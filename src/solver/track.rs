//! Tracking-mode plate solving: solve using an attitude hint instead of
//! the lost-in-space pattern hash.
//!
//! When the caller provides an [`attitude_hint`](super::SolveConfig::attitude_hint)
//! (typically the previous frame's quaternion), the solver can skip pattern-hash
//! lookup entirely:
//!
//! 1. Query catalog stars within a cone around the hinted boresight.
//! 2. Project them to pixel coordinates using the hint rotation.
//! 3. Match each centroid to its nearest predicted star (within a radius set by
//!    hint uncertainty).
//! 4. If enough unique matches exist, run Wahba SVD for a refined rotation.
//! 5. Hand off to the same verification + WCS refine path used by the LIS solver.
//!
//! This succeeds with as few as 3 matched stars (LIS needs 4) and is robust to
//! pattern-hash failures from sparse / low-SNR fields.

use std::time::Instant;

use numeris::{Matrix3, Vector3};
use tracing::debug;

use crate::{Centroid, Quaternion};

use super::solve::{diagonal_factor, elapsed_ms, find_centroid_matches};
use super::{SolveConfig, SolveResult, SolveStatus, SolverDatabase};

/// Minimum unique correspondences required to attempt the SVD step.
const MIN_HINT_MATCHES: usize = 3;

impl SolverDatabase {
    /// Tracking solve using an attitude hint. See [`SolveConfig::attitude_hint`].
    ///
    /// `star_vectors` is the (possibly aberration-corrected) catalog
    /// unit-vector slice prepared by [`SolverDatabase::solve_from_centroids`]
    /// — the same slice the LIS path matches and refines against.
    ///
    /// Returns a [`SolveResult`] with the same shape as the LIS path. On failure
    /// the status is [`SolveStatus::NoMatch`] (or [`SolveStatus::TooFew`] if there
    /// aren't enough centroids).
    pub(crate) fn solve_with_hint(
        &self,
        preprocessed: &[Centroid],
        star_vectors: &[[f32; 3]],
        config: &SolveConfig,
        hint: &Quaternion,
        t0: Instant,
    ) -> SolveResult {
        let cam = &config.camera_model;
        let parity_flip = cam.parity_flip;
        let parity_sign: f32 = if parity_flip { -1.0 } else { 1.0 };

        // True pinhole pixel scale (1/f) from the camera model — the single
        // source of camera geometry. Zero means the model is the unconfigured
        // placeholder, so a hinted solve is impossible.
        let pixel_scale: f32 = config.pixel_scale();
        if pixel_scale <= 0.0 {
            return SolveResult::failure(SolveStatus::NoMatch, elapsed_ms(t0));
        }
        let fov_rad = config.fov_estimate_rad();

        if preprocessed.len() < MIN_HINT_MATCHES {
            return SolveResult::failure(SolveStatus::TooFew, elapsed_ms(t0));
        }

        // ── Hint geometry ──
        let r_hint = hint.to_rotation_matrix();
        // Boresight in ICRS = R^T * [0,0,1] = third row of R
        let boresight_icrs = Vector3::from_array([r_hint[(2, 0)], r_hint[(2, 1)], r_hint[(2, 2)]]);

        // Cone radius: half-FOV (use diagonal for safety) + hint uncertainty + small margin
        let fov_diagonal = fov_rad * diagonal_factor(config);
        let cone_radius = fov_diagonal / 2.0 + config.hint_uncertainty_rad + 2.0 * pixel_scale;
        let nearby_inds = self.star_catalog.query_indices_from_uvec_cached(
            boresight_icrs,
            cone_radius,
            &self.star_vectors,
        );

        debug!(
            "Tracking: hint cone {:.3}° → {} catalog stars",
            cone_radius.to_degrees(),
            nearby_inds.len()
        );

        if nearby_inds.len() < MIN_HINT_MATCHES {
            return SolveResult::failure(SolveStatus::NoMatch, elapsed_ms(t0));
        }

        // ── Sort centroids by brightness (mirrors LIS path) ──
        let mut sorted_indices: Vec<usize> = (0..preprocessed.len()).collect();
        sorted_indices.sort_by(|&a, &b| {
            let ma = preprocessed[a].mass.unwrap_or(f32::MIN);
            let mb = preprocessed[b].mass.unwrap_or(f32::MIN);
            mb.partial_cmp(&ma).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Trim to verification limit (same as LIS).
        let verification_stars = self.props.verification_stars_per_fov as usize;
        let match_centroid_count = preprocessed.len().min(verification_stars);

        // ── Build centroid unit vectors in the camera frame, parity-applied ──
        let centroid_vectors: Vec<[f32; 3]> = sorted_indices
            .iter()
            .map(|&i| {
                let x = parity_sign * preprocessed[i].x * pixel_scale;
                let y = preprocessed[i].y * pixel_scale;
                let z = 1.0f32;
                let norm = (x * x + y * y + z * z).sqrt();
                [x / norm, y / norm, z / norm]
            })
            .collect();

        // ── Project candidate catalog stars to camera-plane angles via the hint ──
        // Note: r_hint maps ICRS→camera, so cam_v = r_hint * icrs_v.
        let half_w = (config.image_width() as f32 / 2.0 + 4.0) * pixel_scale;
        let half_h = (config.image_height() as f32 / 2.0 + 4.0) * pixel_scale;
        let mut projected: Vec<(usize, f32, f32)> = Vec::with_capacity(nearby_inds.len());
        for &cat_idx in &nearby_inds {
            let sv = &star_vectors[cat_idx];
            let icrs_v = Vector3::from_array([sv[0], sv[1], sv[2]]);
            let cam_v = r_hint * icrs_v;
            if cam_v[2] > 0.0 {
                let cx = cam_v[0] / cam_v[2];
                let cy = cam_v[1] / cam_v[2];
                // Only keep stars geometrically inside the (slightly padded) image
                if cx.abs() <= half_w && cy.abs() <= half_h {
                    projected.push((cat_idx, cx, cy));
                }
            }
        }

        if projected.len() < MIN_HINT_MATCHES {
            return SolveResult::failure(SolveStatus::NoMatch, elapsed_ms(t0));
        }

        // ── Initial centroid → catalog star matching ──
        // Match radius covers (a) hint angular uncertainty and (b) the LIS-equivalent
        // fractional match radius. Whichever is larger.
        let hint_match_radius = (config.hint_uncertainty_rad).max(config.match_radius * fov_rad);

        let initial_matches = find_centroid_matches(
            &centroid_vectors[..match_centroid_count.min(centroid_vectors.len())],
            &projected,
            hint_match_radius,
        );

        debug!(
            "Tracking: initial NN match → {} pairs (radius {:.1}″)",
            initial_matches.len(),
            hint_match_radius.to_degrees() * 3600.0
        );

        if initial_matches.len() < MIN_HINT_MATCHES {
            return SolveResult::failure(SolveStatus::NoMatch, elapsed_ms(t0));
        }

        // ── Wahba SVD on the initial correspondence set ──
        let (rotation_matrix, det_sign_ok) =
            wahba_svd_dynamic(&centroid_vectors, star_vectors, &initial_matches);
        if !det_sign_ok {
            // Parity mismatch — bail (caller may still fall back to LIS).
            return SolveResult::failure(SolveStatus::NoMatch, elapsed_ms(t0));
        }

        // ── Verification (same path as LIS) ──
        let (verify_matches, prob_mismatch) = self.verify_attitude(
            &rotation_matrix,
            &centroid_vectors,
            match_centroid_count,
            fov_rad,
            config,
            star_vectors,
        );

        // Same false-positive probability test as LIS, but without the
        // /num_patterns Bonferroni division (no pattern-hash trials happened).
        if prob_mismatch >= config.match_threshold {
            debug!(
                "Tracking: verification rejected (matches={}, prob={:.2e})",
                verify_matches.len(),
                prob_mismatch
            );
            return SolveResult::failure(SolveStatus::NoMatch, elapsed_ms(t0));
        }

        debug!(
            "Tracking: VERIFIED — {} matches, prob={:.2e}",
            verify_matches.len(),
            prob_mismatch
        );

        // ── WCS refinement + finalization (same path as LIS) ──
        match self.refine_and_finalize(
            &rotation_matrix,
            &verify_matches,
            preprocessed,
            &sorted_indices,
            star_vectors,
            config,
            parity_flip,
            fov_rad,
            pixel_scale as f64,
            match_centroid_count,
            MIN_HINT_MATCHES,
            prob_mismatch,
            t0,
        ) {
            Some(result) => result,
            None => SolveResult::failure(SolveStatus::NoMatch, elapsed_ms(t0)),
        }
    }
}

/// Run Wahba SVD on a dynamic-sized correspondence set.
///
/// `centroid_vectors` is indexed by sorted (brightness) centroid index;
/// `star_vectors` by catalog star index. The match pairs are
/// `(centroid_idx, catalog_star_idx)` in those same index spaces.
///
/// Returns the rotation matrix and a flag indicating whether the determinant
/// is positive (true) or negative (false → likely parity mismatch). A failed
/// SVD (degenerate cross-covariance) returns `(zeros, false)`, which the
/// caller treats as a failed hint.
fn wahba_svd_dynamic(
    centroid_vectors: &[[f32; 3]],
    star_vectors: &[[f32; 3]],
    matches: &[(usize, usize)],
) -> (Matrix3<f32>, bool) {
    if matches.len() < MIN_HINT_MATCHES {
        return (Matrix3::<f32>::zeros(), false);
    }

    // Build the cross-covariance directly (find_rotation_matrix is generic on
    // a const N, which a dynamic match set doesn't have).
    let mut h = numeris::Matrix3::<f64>::zeros();
    for &(cent_idx, cat_idx) in matches {
        let img = &centroid_vectors[cent_idx];
        let cat = &star_vectors[cat_idx];
        let img_v =
            numeris::Vector3::<f64>::from_array([img[0] as f64, img[1] as f64, img[2] as f64]);
        let cat_v =
            numeris::Vector3::<f64>::from_array([cat[0] as f64, cat[1] as f64, cat[2] as f64]);
        h += img_v.outer(&cat_v);
    }

    let Ok(svd) = h.svd() else {
        return (Matrix3::<f32>::zeros(), false);
    };
    let u = svd.u();
    let v_t = svd.vt();
    let r64 = *u * *v_t;
    let r = r64.cast::<f32>();
    let det_ok = r.det() > 0.0;
    (r, det_ok)
}
