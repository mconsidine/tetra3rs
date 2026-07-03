//! Plate solving: given image centroids and an approximate FOV, find the
//! camera pointing direction as a quaternion.
//!
//! Closely follows tetra3's `solve_from_centroids()`:
//! 1. Convert centroids to camera-frame unit vectors.
//! 2. Apply cluster-buster thinning.
//! 3. For each 4-centroid combination (brightest first):
//!    a. Compute edge ratios, look up matching catalog patterns.
//!    b. For each match, estimate rotation via SVD (Wahba problem).
//!    c. Verify by projecting catalog stars and counting matches.
//!    d. Accept if false-positive probability is below threshold.

use std::borrow::Cow;
use std::time::Instant;

use numeris::{Matrix3, Quaternion, Vector3};
use tracing::{debug, warn};

use crate::Centroid;

use super::combinations::BreadthFirstCombinations;
use super::database::separation_for_density;
use super::matching;
use super::pattern::{
    compute_edge_ratios, compute_pattern_key, compute_pattern_key_hash, compute_sorted_edge_angles,
    hash_to_index, sort_pattern_by_centroid_distance, NUM_EDGES, NUM_EDGE_RATIOS, PATTERN_SIZE,
};
use super::wcs_refine;
use super::{
    pixel_scale_from_fov, Solution, SolveConfig, SolveFailure, SolveResult, SolveStatus,
    SolverDatabase,
};

#[cfg(feature = "profile")]
use crate::solver::profiling::{self, buckets};

/// Speed of light in km/s.
pub(super) const C_KM_S: f64 = 299_792.458;

/// Classical stellar aberration: true ICRS unit vector → apparent.
///
/// `beta` = observer barycentric velocity / c (dimensionless, ICRS frame).
/// Formula: `s' = (s + β) / |s + β|`.
///
/// This is the exact classical (non-relativistic) result: a photon with true
/// direction `−s` in the rest frame has apparent velocity `−c·s − v` in a
/// frame moving at `v = c·β`, so it appears to arrive from direction
/// `s + β` (then renormalized to unit length). The relativistic correction
/// is `O(β²)`, giving ~2 mas for Earth's orbital β ≈ 10⁻⁴ — well below
/// plate-solve precision.
pub(super) fn aberration_correct(sv: &[f32; 3], beta: &[f64; 3]) -> [f32; 3] {
    let ax = sv[0] as f64 + beta[0];
    let ay = sv[1] as f64 + beta[1];
    let az = sv[2] as f64 + beta[2];
    let norm = (ax * ax + ay * ay + az * az).sqrt();
    [(ax / norm) as f32, (ay / norm) as f32, (az / norm) as f32]
}

// ── Solve entry point ───────────────────────────────────────────────────────

impl SolverDatabase {
    /// Solve for the camera pointing direction given image centroids.
    ///
    /// Centroids should have the `mass` field populated for brightness sorting.
    /// Centroid (x, y) are in pixel coordinates with (0, 0) at the image center.
    /// +X points right, +Y points down in the image.
    ///
    /// The `SolveConfig`'s camera model supplies all camera geometry: the FOV
    /// estimate (from its focal length and image width), the image dimensions,
    /// optical center, parity, and distortion. Use [`SolveConfig::new`] to
    /// build one from a FOV estimate and image dimensions.
    ///
    /// If `fov_max_error_rad` is set, the solver sweeps FOV values across the range
    /// `[fov_estimate - fov_max_error, fov_estimate + fov_max_error]`, trying the
    /// exact estimate first, then spiraling outward. This makes the solver robust
    /// to uncertain FOV estimates.
    ///
    /// Returns a `SolveResult` with the ICRS→camera quaternion on success.
    pub fn solve_from_centroids(
        &self,
        centroids: &[Centroid],
        config: &SolveConfig,
    ) -> SolveResult {
        let t0 = Instant::now();

        // The `SolveConfig::default()` camera model is a placeholder with a zero
        // image size / focal length; a config left at those defaults yields a
        // degenerate FOV and silently NoMatches. Warn loudly so the cause is
        // visible rather than mysterious.
        let cam = &config.camera_model;
        let focal_ok = cam.focal_length_px.is_finite() && cam.focal_length_px > 0.0;
        if cam.image_width == 0 || cam.image_height == 0 || !focal_ok {
            warn!(
                "camera model appears unconfigured (image {}x{}, focal_length_px {}); \
                 solve will not match — build SolveConfig via new()/with_camera_model()",
                cam.image_width, cam.image_height, cam.focal_length_px
            );
        }

        // ── Aberration correction: build corrected catalog vectors if velocity is set ──
        let star_vecs: Cow<[[f32; 3]]> = match config.observer_velocity_km_s {
            Some(v) => {
                let beta = [v[0] / C_KM_S, v[1] / C_KM_S, v[2] / C_KM_S];
                Cow::Owned(
                    self.star_vectors
                        .iter()
                        .map(|sv| aberration_correct(sv, &beta))
                        .collect(),
                )
            }
            None => Cow::Borrowed(&self.star_vectors),
        };

        // ── Preprocess centroids: subtract CRPIX and undistort (pixel-space, FOV-independent) ──
        // Non-finite inputs (NaN/inf) would quantize to bogus pattern keys and
        // degrade the solve to a silent NoMatch, so drop them here (and any that
        // undistort to non-finite) rather than feed them downstream.
        let n_input = centroids.len();
        let preprocessed: Vec<Centroid> = centroids
            .iter()
            .filter(|c| c.x.is_finite() && c.y.is_finite())
            .map(|c| {
                // Subtract optical center offset
                let cx = c.x as f64 - cam.crpix[0];
                let cy = c.y as f64 - cam.crpix[1];
                // Undistort (distorted observed → ideal pinhole)
                let (ux, uy) = cam.distortion.undistort(cx, cy);
                Centroid {
                    x: ux as f32,
                    y: uy as f32,
                    mass: c.mass,
                    cov: c.cov,
                }
            })
            .filter(|c| c.x.is_finite() && c.y.is_finite())
            .collect();
        if preprocessed.len() < n_input {
            debug!(
                "Dropped {} non-finite centroid(s) before solve",
                n_input - preprocessed.len()
            );
        }
        let working_centroids: &[Centroid] = &preprocessed;

        // ── Tracking-mode shortcut: if a hint is provided, try direct correspondence first ──
        if let Some(ref hint) = config.attitude_hint {
            match self.solve_with_hint(working_centroids, &star_vecs, config, hint, t0) {
                Ok(solution) => {
                    debug!(
                        "Hinted solve succeeded in {:.1} ms ({} matches)",
                        solution.solve_time_ms, solution.num_matches
                    );
                    return Ok(solution);
                }
                Err(fail) => {
                    if config.strict_hint {
                        debug!("Hinted solve failed and strict_hint is set — returning failure");
                        return Err(fail);
                    }
                    debug!("Hinted solve failed; falling back to lost-in-space");
                }
            }
        }

        // Too few centroids to ever form a 4-star pattern. This is
        // FOV-independent, unlike the post-thinning TooFew inside
        // `solve_at_fov`, so it ends the solve outright.
        if working_centroids.len() < PATTERN_SIZE {
            return failure(SolveStatus::TooFew, t0);
        }

        // Sort centroids by brightness. FOV-independent, so computed once for
        // the whole sweep.
        let sorted_indices = sort_indices_by_brightness(working_centroids);

        // Build FOV sweep: exact estimate first, then spiral outward
        let fov_values = build_fov_sweep(
            config.fov_estimate_rad(),
            config.fov_max_error_rad,
            config.match_radius,
        );

        debug!(
            "FOV sweep: {} values from {:.2}° to {:.2}°",
            fov_values.len(),
            fov_values
                .iter()
                .cloned()
                .reduce(f32::min)
                .unwrap_or(0.0)
                .to_degrees(),
            fov_values
                .iter()
                .cloned()
                .reduce(f32::max)
                .unwrap_or(0.0)
                .to_degrees(),
        );

        let mut last_status = SolveStatus::NoMatch;

        for &fov_try in &fov_values {
            // Check timeout
            if let Some(t) = config.solve_timeout_ms {
                if elapsed_ms(t0) > t as f32 {
                    return failure(SolveStatus::Timeout, t0);
                }
            }

            debug!("Trying FOV = {:.3}°", fov_try.to_degrees());
            let result = self.solve_at_fov(
                working_centroids,
                &sorted_indices,
                config,
                fov_try,
                &star_vecs,
                t0,
            );
            match result {
                Ok(solution) => return Ok(solution),
                // TooFew here means cluster-buster thinning left fewer than 4
                // pattern centroids. The thinning separation scales with the
                // FOV being tried, so a different FOV in the sweep may still
                // succeed — keep going.
                Err(fail) => last_status = fail.status,
            }
        }

        failure(last_status, t0)
    }

    /// Attempt a solve at a specific FOV value.
    ///
    /// `sorted_indices` is the brightness-sorted centroid index order, computed
    /// once by the caller (it does not depend on the FOV). The caller also
    /// guarantees at least `PATTERN_SIZE` centroids.
    fn solve_at_fov(
        &self,
        centroids: &[Centroid],
        sorted_indices: &[usize],
        config: &SolveConfig,
        fov_estimate: f32,
        star_vectors: &[[f32; 3]],
        t0: Instant,
    ) -> SolveResult {
        #[cfg(feature = "profile")]
        profiling::count(buckets::FOV_PASS, 1);

        // True pinhole pixel scale (rad/px): ps = 1/f where f = (W/2) / tan(fov/2).
        // Derived from the sweep's FOV value, not the camera model's focal
        // length — each sweep iteration tries a different scale.
        let pixel_scale = if config.image_width() > 0 && fov_estimate > 0.0 {
            pixel_scale_from_fov(config.image_width(), fov_estimate as f64) as f32
        } else {
            0.0
        };

        let num_centroids = sorted_indices.len();

        // ── Compute unit vectors in camera frame ──
        // Centroid (x, y) in pixels → scale to radians → uvec = normalize(x_rad, y_rad, 1)
        // Note: distortion correction (if any) was already applied in solve_from_centroids.
        let centroid_vectors = centroid_unit_vectors(centroids, sorted_indices, pixel_scale, 1.0);

        // Lazily-created x-flipped copy for parity-flipped images.
        // Built on first use, cached for subsequent pattern attempts.
        let mut flipped_vectors: Option<Vec<[f32; 3]>> = None;

        // ── Cluster-buster thinning ──
        // Apply the same separation constraint as database generation to avoid
        // wasting pattern attempts on dense clusters.
        let verification_stars = self.props.verification_stars_per_fov;
        let separation = separation_for_density(fov_estimate, verification_stars);
        let cos_sep = separation.cos();

        let mut keep_for_patterns = vec![false; num_centroids];
        for i in 0..num_centroids {
            let vi = &centroid_vectors[i];
            let mut occupied = false;
            for j in 0..i {
                if keep_for_patterns[j] {
                    let vj = &centroid_vectors[j];
                    let dot = vi[0] * vj[0] + vi[1] * vj[1] + vi[2] * vj[2];
                    if dot > cos_sep {
                        occupied = true;
                        break;
                    }
                }
            }
            if !occupied {
                keep_for_patterns[i] = true;
            }
        }

        let pattern_centroid_inds: Vec<usize> = (0..num_centroids)
            .filter(|&i| keep_for_patterns[i])
            .collect();
        let num_pattern_centroids = pattern_centroid_inds.len();

        debug!(
            "Centroids: {} total, {} for patterns after cluster busting",
            num_centroids, num_pattern_centroids
        );

        if num_pattern_centroids < PATTERN_SIZE {
            return failure(SolveStatus::TooFew, t0);
        }

        // Trim match centroids to verification limit
        let match_centroid_count = num_centroids.min(verification_stars as usize);

        // ── Solver parameters ──
        let p_bins = self.props.pattern_bins;
        // A tolerance below the database's quantization error cannot work
        // (patterns were binned at pattern_max_error), so floor it there.
        let p_max_err = match config.match_max_error {
            Some(user_err) if user_err < self.props.pattern_max_error => {
                debug!(
                    "match_max_error {:.2e} below database pattern_max_error {:.2e}; using the latter",
                    user_err, self.props.pattern_max_error
                );
                self.props.pattern_max_error
            }
            Some(user_err) => user_err,
            None => self.props.pattern_max_error,
        };
        // Ceiling on the tolerance. The candidate-key search enumerates a 5-D
        // Cartesian product of ~(2·err·bins + 1)^5 tuples per star combination;
        // with no cap a large match_max_error (e.g. 0.1 at 250 bins ≈ 345M
        // tuples, ~8 GB) exhausts memory. Bound the per-dimension bin span, but
        // never below the database's own quantization error (the floor above).
        const MAX_KEY_SPAN_BINS: f32 = 16.0;
        let err_ceiling =
            (MAX_KEY_SPAN_BINS / (2.0 * p_bins as f32)).max(self.props.pattern_max_error);
        let p_max_err = if p_max_err > err_ceiling {
            debug!(
                "match_max_error {:.2e} exceeds enumeration ceiling {:.2e} ({} bins); clamping",
                p_max_err, err_ceiling, p_bins
            );
            err_ceiling
        } else {
            p_max_err
        };
        let match_threshold = config.match_threshold / self.props.num_patterns as f64;
        let timeout_ms = config.solve_timeout_ms;

        // Guard against a corrupt or placeholder database. An empty table makes
        // the hash-probe arithmetic below divide by zero; `num_patterns == 0`
        // (with a non-empty table) makes `match_threshold` above `+inf`, so
        // every candidate would pass verification and produce a bogus solution.
        let table_len = self.pattern_catalog.len() as u64;
        if table_len == 0 || self.props.num_patterns == 0 {
            return failure(SolveStatus::NoMatch, t0);
        }

        debug!(
            "Checking up to C({},{}) = {} image patterns",
            num_pattern_centroids,
            PATTERN_SIZE,
            n_choose_k(num_pattern_centroids, PATTERN_SIZE)
        );

        // ── Main solve loop ──
        let mut status = SolveStatus::NoMatch;
        let mut pattern_key_list: Vec<(u32, [u32; NUM_EDGE_RATIOS])> = Vec::new();

        for image_pattern_local in
            BreadthFirstCombinations::<PATTERN_SIZE>::new(&pattern_centroid_inds)
        {
            // Check timeout
            if let Some(t) = timeout_ms {
                if elapsed_ms(t0) > t as f32 {
                    debug!("Timeout after {:.1}ms", elapsed_ms(t0));
                    status = SolveStatus::Timeout;
                    break;
                }
            }

            // Get image pattern vectors
            let image_vecs: [[f32; 3]; 4] = [
                centroid_vectors[image_pattern_local[0]],
                centroid_vectors[image_pattern_local[1]],
                centroid_vectors[image_pattern_local[2]],
                centroid_vectors[image_pattern_local[3]],
            ];

            #[cfg(feature = "profile")]
            profiling::count(buckets::COMBOS, 1);

            // Compute edge angles and ratios
            // (image-side edges: this is exactly what an N×N precomputed
            // pairwise-angle matrix would replace with table lookups.)
            let (edge_angles, image_ratios) = timed!(buckets::IMAGE_EDGES, {
                let ea = compute_sorted_edge_angles(&image_vecs);
                let ir = compute_edge_ratios(&ea);
                (ea, ir)
            });
            let image_largest_edge = edge_angles[NUM_EDGES - 1];

            // Broadened range for pattern key lookup
            let ratio_min: [f32; NUM_EDGE_RATIOS] =
                std::array::from_fn(|i| image_ratios[i] - p_max_err);
            let ratio_max: [f32; NUM_EDGE_RATIOS] =
                std::array::from_fn(|i| image_ratios[i] + p_max_err);

            let image_key = compute_pattern_key(&image_ratios, p_bins);

            // Compute the range of pattern keys to search
            let key_min: [u32; NUM_EDGE_RATIOS] =
                std::array::from_fn(|i| (ratio_min[i] * p_bins as f32).max(0.0) as u32);
            let key_max: [u32; NUM_EDGE_RATIOS] =
                std::array::from_fn(|i| (ratio_max[i] * p_bins as f32).min(p_bins as f32) as u32);

            // Build list of candidate pattern keys, sorted by distance from image_key
            pattern_key_list.clear();
            timed!(buckets::KEY_ENUM, {
                enumerate_key_range(&key_min, &key_max, &image_key, &mut pattern_key_list);
                pattern_key_list.sort_unstable_by_key(|&(dist, _)| dist);
            });

            // Try each candidate pattern key
            for (_, pkey) in &pattern_key_list {
                let pkey_hash = compute_pattern_key_hash(pkey, p_bins);
                let hidx = hash_to_index(pkey_hash, table_len);

                // Pre-filter by 16-bit key hash
                let key_hash16 = (pkey_hash & 0xFFFF) as u16;

                // Walk the hash chain inline (quadratic probing). Generator
                // tables keep load ≤ 0.5 on a prime size, so an empty slot is
                // always reached; the `table_len` cap only bounds the walk on a
                // corrupt/over-full table (which would otherwise loop forever).
                for c in 0u64..table_len {
                    let tidx = ((hidx.wrapping_add(c.wrapping_mul(c))) % table_len) as usize;
                    let entry = self.pattern_catalog.get(tidx);
                    if entry.is_empty() {
                        break; // end of chain
                    }
                    if entry.key_hash != key_hash16 {
                        continue;
                    }

                    #[cfg(feature = "profile")]
                    profiling::count(buckets::CANDIDATES, 1);

                    // FOV consistency check: the catalog pattern's largest edge
                    // should be close to the image pattern's largest edge.
                    let cat_largest = entry.largest_edge;
                    if let Some(fov_err) = config.fov_max_error_rad {
                        // Implied FOV from this match
                        let implied_fov = cat_largest / image_largest_edge * fov_estimate;
                        if (implied_fov - fov_estimate).abs() > fov_err {
                            continue;
                        }
                    }

                    // Full edge-ratio comparison
                    let cat_pat = entry.star_indices;
                    let cat_vecs: [[f32; 3]; 4] = [
                        star_vectors[cat_pat[0] as usize],
                        star_vectors[cat_pat[1] as usize],
                        star_vectors[cat_pat[2] as usize],
                        star_vectors[cat_pat[3] as usize],
                    ];
                    // Catalog-side edges: the analogue of `image_edges`, but
                    // computed per surviving candidate and NOT precomputable
                    // per-image (depends on which catalog pattern matched).
                    let (cat_edges, cat_ratios) = timed!(buckets::CAT_EDGES, {
                        let ce = compute_sorted_edge_angles(&cat_vecs);
                        let cr = compute_edge_ratios(&ce);
                        (ce, cr)
                    });
                    let cat_largest_edge = cat_edges[NUM_EDGES - 1];

                    // Check all edge ratios are within tolerance
                    let ratios_ok = (0..NUM_EDGE_RATIOS)
                        .all(|i| cat_ratios[i] > ratio_min[i] && cat_ratios[i] < ratio_max[i]);
                    if !ratios_ok {
                        continue;
                    }

                    // ── Estimate rotation via SVD ──

                    // Refine FOV estimate from this match
                    let fov = cat_largest_edge / image_largest_edge * fov_estimate;

                    // Sort image pattern by centroid distance (canonical ordering)
                    let mut img_order: [usize; 4] = [0, 1, 2, 3];
                    sort_pattern_by_centroid_distance(&mut img_order, |i| image_vecs[i]);

                    // Catalog pattern is already pre-sorted during database generation.
                    // Build matched vector pairs.
                    let matched_img: [[f32; 3]; 4] =
                        std::array::from_fn(|i| image_vecs[img_order[i]]);
                    let matched_cat: [[f32; 3]; 4] = std::array::from_fn(|i| cat_vecs[i]);

                    #[cfg(feature = "profile")]
                    profiling::count(buckets::RATIO_PASS, 1);

                    // SVD rotation: finds R such that camera_vec ≈ R * icrs_vec.
                    // A degenerate cross-covariance (e.g. duplicate or collinear
                    // centroids) fails the SVD — skip the candidate, don't panic.
                    let Some(mut rotation_matrix) = timed!(
                        buckets::SVD,
                        wahba_rotation(matched_img.iter().zip(matched_cat.iter()))
                    ) else {
                        continue;
                    };

                    // Determine parity from the rotation determinant.
                    // centroid_vectors is never mutated; when parity is needed we use
                    // a lazily-created x-flipped copy for verification matching.
                    let parity_flip;
                    let working_vectors: &[[f32; 3]];
                    if rotation_matrix.det() < 0.0 {
                        // Wrong parity (e.g. FITS image with CDELT1 < 0).
                        parity_flip = true;
                        // Derive the parity-flipped rotation WITHOUT a second SVD.
                        //
                        // Flipping the x-component of every image vector is
                        // img' = D·img with D = diag(-1, 1, 1). `find_rotation_matrix`
                        // builds H = Σ imgᵢ · catᵢᵀ, decomposes H = U·S·Vᵀ, and
                        // returns R = U·Vᵀ. With flipped image vectors
                        //   H' = Σ (D·imgᵢ)·catᵢᵀ = D·H = (D·U)·S·Vᵀ,
                        // a valid SVD because D is orthogonal. Hence
                        //   R' = (D·U)·Vᵀ = D·(U·Vᵀ) = D·R,
                        // i.e. R' is just R with its first ROW negated. Since
                        // R = U·Vᵀ is invariant to the per-singular-vector sign
                        // freedom of the decomposition, this is mathematically
                        // exact and reproduces the second SVD bit-for-bit (up to
                        // f32 rounding of a single extra negation).
                        //
                        // det(R') = det(D)·det(R) = −det(R) > 0 here, so R' is
                        // always a proper rotation; the old "still a reflection →
                        // skip" branch can never trigger and is therefore dropped.
                        rotation_matrix[(0, 0)] = -rotation_matrix[(0, 0)];
                        rotation_matrix[(0, 1)] = -rotation_matrix[(0, 1)];
                        rotation_matrix[(0, 2)] = -rotation_matrix[(0, 2)];
                        // Lazily create flipped centroid vectors for matching
                        let fv = flipped_vectors.get_or_insert_with(|| {
                            centroid_vectors
                                .iter()
                                .map(|v| [-v[0], v[1], v[2]])
                                .collect()
                        });
                        working_vectors = fv;
                    } else {
                        parity_flip = false;
                        working_vectors = &centroid_vectors;
                    }

                    // ── Verify by matching nearby catalog stars ──
                    let (current_matches, prob_mismatch) = self.verify_attitude(
                        &rotation_matrix,
                        working_vectors,
                        match_centroid_count,
                        fov,
                        config,
                        star_vectors,
                    );

                    if prob_mismatch >= match_threshold {
                        continue;
                    }

                    debug!(
                        "MATCH: {} matches, prob={:.2e}, fov={:.3}°",
                        current_matches.len(),
                        prob_mismatch * self.props.num_patterns as f64,
                        fov.to_degrees()
                    );

                    // ── WCS TAN-projection refinement ──
                    // The refinement locks its pixel scale to the
                    // pattern-match refined FOV, NOT the camera model's focal
                    // length — deliberately asymmetric with the tracking path
                    // (which trusts the model's 1/f). Lost-in-space must stay
                    // robust to a wrong focal-length estimate; the pattern
                    // match measures the true scale, and the model's f is only
                    // a search seed here. Fewer than 4 surviving matches → try
                    // next candidate.
                    if let Some(result) = self.refine_and_finalize(
                        &rotation_matrix,
                        &current_matches,
                        centroids,
                        sorted_indices,
                        star_vectors,
                        config,
                        parity_flip,
                        fov,
                        pixel_scale_from_fov(config.image_width(), fov as f64),
                        match_centroid_count,
                        4,
                        prob_mismatch * self.props.num_patterns as f64,
                        t0,
                    ) {
                        return Ok(result);
                    }
                }
            }
        }

        failure(status, t0)
    }

    /// Verify a candidate attitude by projecting nearby catalog stars into
    /// the camera frame and greedily matching them to image centroids.
    ///
    /// Shared by the lost-in-space and tracking paths. `centroid_vectors`
    /// must be brightness-sorted with parity already applied; `star_vectors`
    /// is the (possibly aberration-corrected) catalog unit-vector slice. The
    /// cone query itself uses the stored raw vectors, which are bit-identical
    /// to `Star::uvec()` and aligned with the catalog, so the candidate set
    /// is unchanged by aberration.
    ///
    /// Returns the matches `(centroid_local_idx, catalog_star_idx)` and the
    /// binomial false-positive probability of the match count (before any
    /// Bonferroni correction for the number of pattern trials).
    pub(super) fn verify_attitude(
        &self,
        rotation_matrix: &Matrix3<f32>,
        centroid_vectors: &[[f32; 3]],
        match_centroid_count: usize,
        fov: f32,
        config: &SolveConfig,
        star_vectors: &[[f32; 3]],
    ) -> (Vec<(usize, usize)>, f64) {
        let fov_diagonal = fov * diagonal_factor(config);
        let match_radius_rad = config.match_radius * fov;

        // Find catalog stars within the diagonal FOV
        let image_center_icrs = rotation_matrix.transpose() * Vector3::from_array([0.0, 0.0, 1.0]);
        let nearby_inds = timed!(
            buckets::VERIFY_QUERY,
            self.star_catalog.query_indices_from_uvec_cached(
                image_center_icrs,
                fov_diagonal / 2.0,
                &self.star_vectors,
            )
        );
        #[cfg(feature = "profile")]
        profiling::count(buckets::VERIFY_QUERY_STARS, nearby_inds.len() as u64);

        // Project catalog stars to camera frame; keep stars in front (z > 0).
        let mut nearby_cam_positions: Vec<(usize, f32, f32)> = Vec::new();
        for &cat_idx in &nearby_inds {
            let sv = &star_vectors[cat_idx];
            let icrs_v = Vector3::from_array([sv[0], sv[1], sv[2]]);
            let cam_v = *rotation_matrix * icrs_v;
            if cam_v[2] > 0.0 {
                nearby_cam_positions.push((cat_idx, cam_v[0] / cam_v[2], cam_v[1] / cam_v[2]));
            }
        }
        // Limit to 2x the number of image centroids (like tetra3)
        nearby_cam_positions.truncate(2 * match_centroid_count);
        let num_nearby = nearby_cam_positions.len();

        // Match image centroids to projected catalog stars
        let matches = timed!(
            buckets::VERIFY_MATCH,
            find_centroid_matches(
                &centroid_vectors[..match_centroid_count.min(centroid_vectors.len())],
                &nearby_cam_positions,
                match_radius_rad,
            )
        );

        // False-positive probability of this match count
        let prob_single = num_nearby as f64 * (config.match_radius as f64).powi(2);
        let prob_mismatch = binomial_cdf(
            (match_centroid_count as i64 - (matches.len() as i64 - 2)).max(0) as u32,
            match_centroid_count as u32,
            1.0 - prob_single.min(1.0),
        );

        (matches, prob_mismatch)
    }

    /// Run the WCS refinement on a verified match set and assemble the final
    /// [`SolveResult`].
    ///
    /// Shared by the lost-in-space and tracking paths: builds the
    /// parity-applied pixel coordinate list, runs the constrained WCS
    /// refinement, and finalizes. Returns `None` when refinement keeps fewer
    /// than `min_matches` stars — LIS treats that as "try the next
    /// candidate"; tracking treats it as a failed hint.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn refine_and_finalize(
        &self,
        rotation_matrix: &Matrix3<f32>,
        verify_matches: &[(usize, usize)],
        centroids: &[Centroid],
        sorted_indices: &[usize],
        star_vectors: &[[f32; 3]],
        config: &SolveConfig,
        parity_flip: bool,
        fov: f32,
        pixel_scale: f64,
        match_centroid_count: usize,
        min_matches: usize,
        prob: f64,
        t0: Instant,
    ) -> Option<Solution> {
        // Build pixel coordinates: centroids are already CRPIX-subtracted and
        // undistorted. Apply the detected parity.
        let parity_sign: f64 = if parity_flip { -1.0 } else { 1.0 };
        let centroids_px: Vec<(f64, f64)> = sorted_indices
            .iter()
            .map(|&i| (parity_sign * centroids[i].x as f64, centroids[i].y as f64))
            .collect();

        let match_radius_rad = config.match_radius * fov;

        #[cfg(feature = "profile")]
        profiling::count(buckets::WCS_REFINE, 1);
        let wcs_result = timed!(
            buckets::WCS_REFINE,
            wcs_refine::wcs_refine(
                rotation_matrix,
                verify_matches,
                &centroids_px,
                star_vectors,
                &self.star_catalog,
                pixel_scale,
                parity_flip,
                match_radius_rad,
                match_centroid_count,
                10,
            )
        );

        if wcs_result.matches.len() < min_matches {
            return None;
        }

        Some(self.finalize_solve_result(
            &wcs_result,
            star_vectors,
            sorted_indices,
            &centroids_px,
            config,
            parity_flip,
            prob,
            t0,
        ))
    }

    /// Assemble a [`Solution`] from a completed WCS refinement.
    ///
    /// Shared by the lost-in-space (`solve_at_fov`) and tracking
    /// (`solve_with_hint`) paths. `star_vectors` is the (possibly
    /// aberration-corrected) catalog unit-vector slice; `prob` is the caller's
    /// false-positive probability estimate. The match set, residual statistics,
    /// quaternion, and camera model are derived from `wcs_result`.
    #[allow(clippy::too_many_arguments)]
    fn finalize_solve_result(
        &self,
        wcs_result: &wcs_refine::WcsRefineResult,
        star_vectors: &[[f32; 3]],
        sorted_indices: &[usize],
        centroids_px: &[(f64, f64)],
        config: &SolveConfig,
        parity_flip: bool,
        prob: f64,
        t0: Instant,
    ) -> Solution {
        // Derive the rotation directly from the constrained-fit parameters
        // (θ, CRVAL). The pixel scale was locked during refinement, so it is
        // the exact scale of the solution — no CD-matrix decomposition
        // needed. θ describes the parity-applied working frame, so the
        // rotation is proper regardless of `parity_flip`; the residual loop
        // below consistently uses the parity-applied `centroids_px`.
        let refined_rotation = wcs_refine::rotation_from_theta_crval(
            wcs_result.theta_rad,
            wcs_result.crval_rad[0],
            wcs_result.crval_rad[1],
        );
        let ps = wcs_result.pixel_scale as f32;
        let refined_fov =
            (2.0 * ((wcs_result.pixel_scale * config.image_width() as f64) / 2.0).atan()) as f32;

        // Build matched catalog IDs, centroid indices, and angular residuals.
        let mut matched_cat_ids: Vec<i64> = Vec::with_capacity(wcs_result.matches.len());
        let mut matched_cent_inds: Vec<usize> = Vec::with_capacity(wcs_result.matches.len());
        let mut angular_residuals: Vec<f32> = Vec::with_capacity(wcs_result.matches.len());
        for &(cent_local_idx, cat_star_idx) in &wcs_result.matches {
            matched_cat_ids.push(self.star_catalog_ids[cat_star_idx]);
            matched_cent_inds.push(sorted_indices[cent_local_idx]);
            // Compute angular residual using rotation matrix
            let (px, py) = centroids_px[cent_local_idx];
            let ix = px as f32 * ps;
            let iy = py as f32 * ps;
            let iz = 1.0f32;
            let norm = (ix * ix + iy * iy + iz * iz).sqrt();
            let img_v = refined_rotation.transpose()
                * Vector3::from_array([ix / norm, iy / norm, iz / norm]);
            let sv = &star_vectors[cat_star_idx];
            let cat_v = Vector3::from_array([sv[0], sv[1], sv[2]]);
            let cross = img_v.cross(&cat_v);
            let ang = cross.norm().atan2(img_v.dot(&cat_v));
            angular_residuals.push(ang);
        }
        angular_residuals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let rmse = if angular_residuals.is_empty() {
            0.0
        } else {
            (angular_residuals.iter().map(|r| r * r).sum::<f32>() / angular_residuals.len() as f32)
                .sqrt()
        };
        let p90e = if angular_residuals.is_empty() {
            0.0
        } else {
            angular_residuals[(0.9 * (angular_residuals.len() - 1) as f32) as usize]
        };
        let max_err = angular_residuals.last().copied().unwrap_or(0.0);

        // Convert rotation to quaternion
        let quat = Quaternion::from_rotation_matrix(&refined_rotation);

        // Build result camera model: copy the input model (which carries the
        // image dimensions, CRPIX, and distortion), then update the focal
        // length from the refinement's locked pixel scale and record the
        // detected parity.
        let mut result_cam = config.camera_model.clone();
        result_cam.focal_length_px = 1.0 / wcs_result.pixel_scale;
        result_cam.parity_flip = parity_flip;

        Solution {
            qicrs2cam: quat,
            fov_rad: refined_fov,
            num_matches: wcs_result.matches.len() as u32,
            rmse_rad: rmse,
            p90e_rad: p90e,
            max_err_rad: max_err,
            prob,
            solve_time_ms: elapsed_ms(t0),
            parity_flip,
            matched_catalog_ids: matched_cat_ids,
            matched_centroid_indices: matched_cent_inds,
            cd_matrix: wcs_result.cd_matrix,
            crval_rad: wcs_result.crval_rad,
            camera_model: result_cam,
            theta_rad: wcs_result.theta_rad,
        }
    }
}

// ── Helper functions ────────────────────────────────────────────────────────

/// Build FOV values to try: exact estimate first, then spiraling outward.
///
/// Step size is chosen so that the verification match_radius can tolerate the
/// worst-case scale error at the midpoint between steps.
fn build_fov_sweep(fov_estimate: f32, fov_max_error: Option<f32>, match_radius: f32) -> Vec<f32> {
    let mut values = vec![fov_estimate];

    if let Some(max_error) = fov_max_error {
        if max_error > 0.0 {
            // Step = 4 * match_radius * fov_estimate.
            // At the midpoint between steps the relative scale error is
            // step/(2·fov) = 2·mr. A star at the field edge then has position
            // error ≈ 2·mr · (fov/2) = mr·fov — right at the verification
            // match radius (match_radius_rad = mr·fov) — and every star
            // inboard of the edge proportionally less, so verification still
            // matches most of the field. Measured on the profile harness
            // (10° FOV, mr = 0.01, `T3_FOV_BIAS`): a 2% scale error solves
            // 100% of fields and 3% solves 99%, so the 2·mr = 2% midpoint
            // keeps full solve rate while halving the sweep length (the sweep
            // multiplies no-match / wrong-FOV latency).
            let step = (4.0 * match_radius * fov_estimate).max(0.001_f32.to_radians());
            let mut offset = step;
            while offset <= max_error {
                values.push(fov_estimate + offset);
                if fov_estimate - offset > 0.0 {
                    values.push(fov_estimate - offset);
                }
                offset += step;
            }
        }
    }

    values
}

pub(super) fn elapsed_ms(t0: Instant) -> f32 {
    t0.elapsed().as_secs_f32() * 1000.0
}

/// Build a failed [`SolveResult`] with the elapsed time since `t0`.
pub(super) fn failure(status: SolveStatus, t0: Instant) -> SolveResult {
    Err(SolveFailure {
        status,
        solve_time_ms: elapsed_ms(t0),
    })
}

/// Ratio of the image diagonal to the image width, used to size the
/// verification cone (`fov_diagonal = fov * factor`).
///
/// At least 1.42 (≳ √2, the historical square-image constant, kept as a
/// conservative floor); larger for portrait images where the height exceeds
/// the width-referenced FOV and √2 would under-query the corners.
pub(super) fn diagonal_factor(config: &SolveConfig) -> f32 {
    let aspect = config.image_height() as f32 / config.image_width().max(1) as f32;
    (1.0 + aspect * aspect).sqrt().max(1.42)
}

fn n_choose_k(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let mut result = 1usize;
    for i in 0..k {
        // Saturate rather than overflow-panic for very large centroid counts;
        // this value only feeds a debug log, so a saturated estimate is fine.
        result = result.saturating_mul(n - i) / (i + 1);
    }
    result
}

/// Enumerate all pattern keys in the given range, tagged with distance² from center.
fn enumerate_key_range(
    key_min: &[u32; NUM_EDGE_RATIOS],
    key_max: &[u32; NUM_EDGE_RATIOS],
    center: &[u32; NUM_EDGE_RATIOS],
    out: &mut Vec<(u32, [u32; NUM_EDGE_RATIOS])>,
) {
    // Recursive Cartesian product over the 5 dimensions.
    let mut current = [0u32; NUM_EDGE_RATIOS];
    enumerate_key_range_recursive(key_min, key_max, center, 0, &mut current, out);
}

fn enumerate_key_range_recursive(
    key_min: &[u32; NUM_EDGE_RATIOS],
    key_max: &[u32; NUM_EDGE_RATIOS],
    center: &[u32; NUM_EDGE_RATIOS],
    dim: usize,
    current: &mut [u32; NUM_EDGE_RATIOS],
    out: &mut Vec<(u32, [u32; NUM_EDGE_RATIOS])>,
) {
    if dim == NUM_EDGE_RATIOS {
        let dist_sq: u32 = (0..NUM_EDGE_RATIOS)
            .map(|i| {
                let d = current[i] as i32 - center[i] as i32;
                (d * d) as u32
            })
            .sum();
        out.push((dist_sq, *current));
        return;
    }
    for v in key_min[dim]..=key_max[dim] {
        current[dim] = v;
        enumerate_key_range_recursive(key_min, key_max, center, dim + 1, current, out);
    }
}

/// Brightness-sorted centroid index order: highest mass (brightest) first,
/// centroids without mass last. Shared by the LIS and tracking front-ends.
pub(super) fn sort_indices_by_brightness(centroids: &[Centroid]) -> Vec<usize> {
    let mut sorted_indices: Vec<usize> = (0..centroids.len()).collect();
    sorted_indices.sort_by(|&a, &b| {
        let ma = centroids[a].mass.unwrap_or(f32::MIN);
        let mb = centroids[b].mass.unwrap_or(f32::MIN);
        mb.partial_cmp(&ma).unwrap_or(std::cmp::Ordering::Equal)
    });
    sorted_indices
}

/// Camera-frame unit vectors for brightness-ordered centroids:
/// `normalize(parity·x·ps, y·ps, 1)`. The LIS path passes `parity_sign = 1.0`
/// (it detects parity later from the rotation determinant); tracking applies
/// the camera model's parity up front.
pub(super) fn centroid_unit_vectors(
    centroids: &[Centroid],
    sorted_indices: &[usize],
    pixel_scale: f32,
    parity_sign: f32,
) -> Vec<[f32; 3]> {
    sorted_indices
        .iter()
        .map(|&i| {
            let x = parity_sign * centroids[i].x * pixel_scale;
            let y = centroids[i].y * pixel_scale;
            let z = 1.0f32;
            let norm = (x * x + y * y + z * z).sqrt();
            [x / norm, y / norm, z / norm]
        })
        .collect()
}

/// Compute the least-squares rotation matrix from paired image/catalog unit
/// vectors (Wahba's problem).
///
/// Uses SVD of the cross-covariance matrix H = Σ(img_i ⊗ cat_i).
/// The resulting R satisfies: camera_vec ≈ R * icrs_vec.
///
/// The SVD is computed in f64 for precision, then the result is converted back
/// to f32. Returns `None` if the SVD fails (degenerate cross-covariance from
/// pathological input vectors). Serves both the fixed-size 4-star LIS pattern
/// and the tracking path's dynamic correspondence sets.
pub(super) fn wahba_rotation<'a>(
    pairs: impl IntoIterator<Item = (&'a [f32; 3], &'a [f32; 3])>,
) -> Option<Matrix3<f32>> {
    let mut h = numeris::Matrix3::<f64>::zeros();
    for (img, cat) in pairs {
        let img_v =
            numeris::Vector3::<f64>::from_array([img[0] as f64, img[1] as f64, img[2] as f64]);
        let cat_v =
            numeris::Vector3::<f64>::from_array([cat[0] as f64, cat[1] as f64, cat[2] as f64]);
        h += img_v.outer(&cat_v);
    }

    let svd = h.svd().ok()?;
    let u = svd.u();
    let v_t = svd.vt();
    let r64 = *u * *v_t;
    Some(r64.cast::<f32>())
}

/// Find unique 1-to-1 matches between image centroids and projected catalog positions.
///
/// Returns Vec<(centroid_local_idx, catalog_star_idx)>.
pub(super) fn find_centroid_matches(
    centroid_vectors: &[[f32; 3]],
    catalog_positions: &[(usize, f32, f32)], // (star_idx, cam_x, cam_y) in radians
    match_radius: f32,
) -> Vec<(usize, usize)> {
    // For each centroid, project to camera-plane angular coordinates
    let centroid_xy: Vec<(f32, f32)> = centroid_vectors
        .iter()
        .map(|v| {
            if v[2] > 0.0 {
                (v[0] / v[2], v[1] / v[2])
            } else {
                (f32::MAX, f32::MAX)
            }
        })
        .collect();

    let mut scratch = matching::MatchScratch::<f32>::default();
    matching::greedy_unique_matches(
        &centroid_xy,
        centroid_xy.len(),
        catalog_positions,
        match_radius * match_radius,
        &mut scratch,
    )
    .to_vec()
}

// ── Binomial CDF (no external dependency) ───────────────────────────────────

/// Compute the binomial CDF: P(X <= k) where X ~ Binomial(n, p).
/// Uses iterative computation for numerical stability at typical sizes (n < 500).
pub(super) fn binomial_cdf(k: u32, n: u32, p: f64) -> f64 {
    if k >= n {
        return 1.0;
    }
    if p <= 0.0 {
        return 1.0;
    }
    if p >= 1.0 {
        return 0.0; // k < n here (k >= n already returned above)
    }

    let q = 1.0 - p;

    // Start with P(X=0) = q^n, then iteratively compute P(X=i)
    let mut cdf = 0.0;
    let mut log_term = n as f64 * q.ln(); // log(P(X=0))
    cdf += log_term.exp();

    for i in 1..=k as u64 {
        log_term += ((n as u64 - i + 1) as f64).ln() - (i as f64).ln() + p.ln() - q.ln();
        cdf += log_term.exp();
    }

    cdf.min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aberration_correct_shift_direction() {
        // Star along +X, velocity 30 km/s along +Y
        // Aberration should shift the apparent position toward +Y
        let star = [1.0f32, 0.0, 0.0];
        let beta = [0.0, 30.0 / C_KM_S, 0.0];
        let apparent = aberration_correct(&star, &beta);

        // Output should be normalized
        let norm = (apparent[0] as f64 * apparent[0] as f64
            + apparent[1] as f64 * apparent[1] as f64
            + apparent[2] as f64 * apparent[2] as f64)
            .sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "output not unit length: {norm}");

        // Y component should be positive (shifted toward velocity direction)
        assert!(
            apparent[1] > 0.0,
            "expected positive Y shift, got {}",
            apparent[1]
        );

        // Shift magnitude should be ~v/c ≈ 1e-4 rad ≈ 20"
        let shift_rad = (apparent[1] as f64).atan2(apparent[0] as f64);
        let expected = 30.0 / C_KM_S; // ~1e-4 rad
        assert!(
            (shift_rad - expected).abs() < 1e-6,
            "shift {shift_rad:.2e} rad, expected ~{expected:.2e} rad"
        );
    }

    #[test]
    fn test_aberration_correct_zero_velocity() {
        // Zero velocity should return the original unit vector unchanged
        let s = 1.0f32 / 3.0f32.sqrt();
        let star = [s, s, s];
        let beta = [0.0, 0.0, 0.0];
        let apparent = aberration_correct(&star, &beta);
        for i in 0..3 {
            assert!(
                (apparent[i] - star[i]).abs() < 1e-6,
                "component {i} changed: {} -> {}",
                star[i],
                apparent[i]
            );
        }
    }

    #[test]
    fn test_aberration_correct_parallel_velocity() {
        // Velocity parallel to star direction should produce zero transverse shift
        let star = [1.0f32, 0.0, 0.0];
        let beta = [30.0 / C_KM_S, 0.0, 0.0];
        let apparent = aberration_correct(&star, &beta);

        // Y and Z should remain essentially zero
        assert!(apparent[1].abs() < 1e-7, "Y not zero: {}", apparent[1]);
        assert!(apparent[2].abs() < 1e-7, "Z not zero: {}", apparent[2]);
        // X should still be ~1.0 (normalized)
        assert!((apparent[0] - 1.0).abs() < 1e-6);
    }
}
