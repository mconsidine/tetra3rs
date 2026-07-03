//! Default connected-component-labeling extraction path: local background
//! subtraction, sigma-clipped global noise stats, optional matched filter,
//! threshold → CCL → two-pass per-blob moments with annulus background, and
//! sub-pixel peak refinement. Split out of the crate-facing module; entry via
//! [`extract_from_gray`].

use std::borrow::Cow;

use numeris::imageproc::{
    connected_components_with_label_buffer, gaussian_blur, BorderMode, Component, Connectivity,
};
use numeris::DynMatrix;

use super::{
    accepted_peak_refine, elongation_from_cov, median_f32, par, peak_sharpness,
    sort_and_truncate_by_mass, BackgroundGrid, CentroidExtractionConfig, CentroidExtractionResult,
};
use crate::centroid::Centroid;
use crate::error::{Error, Result};

/// Shared extraction pipeline for both image and raw-pixel entry points.
pub(super) fn extract_from_gray(
    gray_input: &[f32],
    width: u32,
    height: u32,
    config: &CentroidExtractionConfig,
) -> Result<CentroidExtractionResult> {
    let w = width as usize;
    let h = height as usize;

    // ── Step 0: validate geometry and config ──
    // The pipeline below indexes `width - 1` and chunks the image into rows of
    // width `w`, both of which panic on a degenerate image; and a zero
    // `local_bg_block_size` divides by zero in the background estimator. Reject
    // these up front (the fast path guards the same cases).
    if w < 2 || h < 2 {
        return Err(Error::InvalidInput(format!(
            "image must be at least 2x2, got {width}x{height}"
        )));
    }
    if config.local_bg_block_size == Some(0) {
        return Err(Error::InvalidInput(
            "local_bg_block_size must be >= 1 (or None)".into(),
        ));
    }
    if !config.sigma_threshold.is_finite() {
        return Err(Error::InvalidInput(format!(
            "sigma_threshold must be finite, got {}",
            config.sigma_threshold
        )));
    }

    let filter_sigma = config
        .matched_filter_sigma
        .filter(|s| s.is_finite() && *s > 0.0);

    // ── Steps 1-2: background model, residuals, and noise stats ──
    // With `local_bg_block_size` set, a block-median grid is built (from a
    // staggered subsample) and everything downstream works from residuals
    // against its bilinear surface. The residuals are produced by ONE fused
    // pass over the image that interpolates the surface on the fly and
    // writes the clamped measurement image and — only when the matched
    // filter is on — the unclamped filter input directly into the blur's
    // matrix. (Blurring the clamped image would rectify negative noise into
    // a positive DC offset; measuring on the unclamped one would let
    // negative pixels cancel star flux.) The materialized full-image
    // background buffer and its separate subtract passes are gone.
    //
    // Noise statistics use the same estimator either way; on the local-bg
    // path it runs on bilinear residuals at the block subsample lattice
    // (identical reference surface, ~stride² fewer samples) instead of a
    // full-image residual buffer.
    let gray: Cow<[f32]>;
    let bg_mean: f32;
    let bg_sigma: f32;
    let mut filter_input: Option<DynMatrix<f32>> = None;

    if let Some(block_size) = config.local_bg_block_size {
        let bs = block_size as usize;
        // Stride bs/16 (vs the fast path's bs/8) keeps extra sampling margin
        // on this calibration-quality path; the build-time σ is ignored in
        // favor of the estimator below, run against the bilinear surface.
        let (bg, _) = BackgroundGrid::build(gray_input, w, h, bs, (bs / 16).max(1));

        let residuals = subsample_residuals(gray_input, w, h, &bg);
        (bg_mean, bg_sigma) = estimate_background(&residuals, width, height, config);

        // Fused residual pass (rows in parallel under the `parallel` feature;
        // each row writes disjoint output, results independent of threads).
        let mut clamped = vec![0.0_f32; w * h];
        if filter_sigma.is_some() {
            let mut unclamped = vec![0.0_f32; w * h];
            par::for_each_chunk_pair_mut(&mut clamped, &mut unclamped, w, |y, cr, ur| {
                let rp = bg.row_params(y);
                let row = y * w;
                for x in 0..w {
                    let r = gray_input[row + x] - bg.value_at(x, rp);
                    cr[x] = r.max(0.0);
                    ur[x] = r;
                }
            });
            filter_input = Some(DynMatrix::from_vec(w, h, unclamped));
        } else {
            par::for_each_chunk_mut(&mut clamped, w, |y, cr| {
                let rp = bg.row_params(y);
                let row = y * w;
                for (x, out) in cr.iter_mut().enumerate() {
                    *out = (gray_input[row + x] - bg.value_at(x, rp)).max(0.0);
                }
            });
        }
        gray = Cow::Owned(clamped);
    } else {
        (bg_mean, bg_sigma) = estimate_background(gray_input, width, height, config);
        gray = Cow::Borrowed(gray_input);
        if filter_sigma.is_some() {
            filter_input = Some(DynMatrix::from_vec(w, h, gray_input.to_vec()));
        }
    }
    let gray: &[f32] = &gray;

    // ── Step 3: optional matched filter for thresholding only ──
    // The unclamped residual is convolved with a Gaussian and threshold/CCL
    // run on the filtered copy; centroids are still measured on the
    // unfiltered `gray`, so intensities and CoM positions are unaffected.
    // The detection threshold is scaled by the kernel's white-noise
    // suppression factor so `sigma_threshold` keeps meaning "sigmas of the
    // noise actually present in the thresholded image", filter on or off.
    // Under the `parallel` feature numeris's gaussian_blur runs
    // multi-threaded.
    let (thresh_src, mask_threshold): (Cow<[f32]>, f32) = match (filter_sigma, filter_input) {
        (Some(sigma), Some(mat)) => {
            let filtered = gaussian_blur(&mat, sigma, BorderMode::Replicate).into_vec();
            let suppression = gaussian_noise_suppression(sigma);
            (
                Cow::Owned(filtered),
                bg_mean + config.sigma_threshold * bg_sigma * suppression,
            )
        }
        _ => (
            Cow::Borrowed(gray),
            bg_mean + config.sigma_threshold * bg_sigma,
        ),
    };
    let thresh_src: &[f32] = &thresh_src;

    // ── Step 4: threshold and label blobs ──
    // Build a u8 mask DynMatrix in proper (h, w) layout for CCL — its
    // bbox/labels conventions assume the supplied dimensions match the image.
    let mask = DynMatrix::<u8>::from_fn(h, w, |r, c| {
        if thresh_src[r * w + c] > mask_threshold {
            1u8
        } else {
            0u8
        }
    });
    let connectivity = if config.use_8_connectivity {
        Connectivity::Eight
    } else {
        Connectivity::Four
    };
    let (labels, components) = connected_components_with_label_buffer(&mask, connectivity, 0u8);

    // ── Step 4: compute centroids ──
    let raw_centroids = compute_blob_centroids(gray, &labels, &components, width, height, config);
    // "Raw" blob count = connected components before the size/elongation/mass
    // filters, matching the field's documented meaning and the fast path's
    // pre-`min_pixels` region count.
    let num_blobs_raw = components.len();

    // ── Step 5: convert to centered pixel coordinates ──
    // Origin at the geometric image center, (W-1)/2 and (H-1)/2 (pixel centers
    // are at integer indices, so for even dimensions this is the intersection
    // of the four central pixels — matching the FITS / astropy / OpenCV
    // convention). +X right, +Y down.
    let cx = (width - 1) as f32 / 2.0;
    let cy = (height - 1) as f32 / 2.0;

    let mut centroids: Vec<Centroid> = raw_centroids
        .into_iter()
        .map(|rc| Centroid {
            x: rc.x_px - cx,
            y: rc.y_px - cy,
            mass: Some(rc.mass),
            cov: Some(rc.cov),
        })
        .collect();

    sort_and_truncate_by_mass(&mut centroids, config.max_centroids);

    Ok(CentroidExtractionResult {
        centroids,
        image_width: width,
        image_height: height,
        background_mean: bg_mean,
        background_sigma: bg_sigma,
        threshold: mask_threshold,
        num_blobs_raw,
    })
}

/// Background-subtracted residuals at the block subsample lattice (the same
/// staggered lattice [`BackgroundGrid::build`] medians over), against the
/// bilinear surface — the identical reference the full-image residual pass
/// used, so feeding these to [`estimate_background`] preserves its semantics
/// while touching ~stride² fewer samples.
fn subsample_residuals(pixels: &[f32], w: usize, h: usize, bg: &BackgroundGrid) -> Vec<f32> {
    let stride = bg.stride();
    let mut out: Vec<f32> = Vec::with_capacity((w / stride + 1) * (h / stride + 1));
    let mut y = 0usize;
    let mut phase = 0usize;
    while y < h {
        let rp = bg.row_params(y);
        let row = y * w;
        let mut x = phase;
        while x < w {
            let v = pixels[row + x];
            if v.is_finite() {
                out.push(v - bg.value_at(x, rp));
            }
            x += stride;
        }
        phase = (phase + 1) % stride;
        y += stride;
    }
    out
}

/// White-noise standard-deviation suppression factor of the separable 2-D
/// Gaussian blur used by the matched filter.
///
/// For a normalized 1-D kernel `k`, convolving white noise multiplies its
/// standard deviation by `√(Σk²)` per axis, so the separable 2-D factor is
/// `Σk²`. The kernel is replicated exactly as numeris builds it (radius
/// `ceil(3σ)`, `exp(−x²/2σ²)` weights, normalized); for σ ≳ 1 this
/// approaches the continuous limit `1/(2√π·σ)`.
fn gaussian_noise_suppression(sigma: f32) -> f32 {
    let radius = (3.0 * sigma).ceil() as i64;
    let inv_two_sigma_sq = 1.0 / (2.0 * sigma as f64 * sigma as f64);
    let mut sum = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    for i in -radius..=radius {
        let w = (-((i * i) as f64) * inv_two_sigma_sq).exp();
        sum += w;
        sum_sq += w * w;
    }
    (sum_sq / (sum * sum)) as f32
}

/// Estimate background level and noise.
///
/// Uses the median as the background level and estimates noise from the
/// lower half of the pixel distribution (below the median). This is robust
/// to contamination from stars and nebulosity, which only bias upward.
///
/// The noise estimate sigma-clips the below-median tail to reject remaining
/// outliers, then mirrors the lower-half RMS **about the median** to get the
/// full Gaussian sigma (`E[(v−m)² | v ≤ m] = σ²`).
pub(super) fn estimate_background(
    gray: &[f32],
    _width: u32,
    _height: u32,
    config: &CentroidExtractionConfig,
) -> (f32, f32) {
    let mut values: Vec<f32> = gray.iter().copied().filter(|v| v.is_finite()).collect();
    if values.is_empty() {
        return (0.0, 0.0);
    }

    // Median as robust background level (O(n) selection; see `median_f32`).
    let median = median_f32(&mut values);

    // Estimate noise from pixels at or below the median (uncontaminated by
    // stars, which only push the distribution upward). For Gaussian noise the
    // second moment of the lower half about the *median* equals the full
    // variance — E[(v−m)² | v ≤ m] = σ² — so the lower-half RMS about the
    // median mirrors directly into the full Gaussian sigma. This matches the
    // fast path's `coarse_background`. (Historically this computed the RMS
    // about the lower half's own mean, which for a half-normal is only
    // ≈0.60σ — silently turning a nominal 5σ threshold into a ~3σ one.)
    let mut low_half: Vec<f32> = values.iter().copied().filter(|&v| v <= median).collect();

    // Sigma-clip the lower tail to reject remaining outliers (dead or
    // negative pixels), re-estimating about the median each pass.
    let mut sigma = 0.0_f32;
    for _ in 0..config.sigma_clip_iterations {
        if low_half.is_empty() {
            break;
        }
        let var_sum: f64 = low_half
            .iter()
            .map(|&v| ((v - median) as f64).powi(2))
            .sum();
        sigma = (var_sum / low_half.len() as f64).sqrt() as f32;
        if sigma < 1e-10 {
            break;
        }
        let lo = median - config.sigma_clip_factor * sigma;
        let before = low_half.len();
        low_half.retain(|&v| v >= lo);
        if low_half.len() == before {
            break; // converged
        }
    }

    (median, sigma)
}

/// Raw pixel-coordinate centroid with mass and covariance.
struct RawCentroid {
    x_px: f32,
    y_px: f32,
    mass: f32,
    /// Intensity-weighted 2×2 covariance matrix [[cxx, cxy], [cxy, cyy]] in pixels².
    cov: crate::Matrix2,
}

/// Compute intensity-weighted centroids for each labeled blob.
///
/// Consumes [`numeris::imageproc::Component`]s for area / bounding box, plus
/// the row-major labels buffer for per-pixel masking. For each blob that
/// passes size and elongation filters:
/// 1. A local background is estimated from the median of non-blob pixels in a
///    5-pixel annulus around the blob's bounding box.
/// 2. Intensity-weighted moments are accumulated with the local background
///    subtracted, yielding a center-of-mass (CoM) position. Peak pixel is
///    tracked in the same pass.
/// 3. A 2D quadratic is fit to the 3×3 neighborhood around the peak pixel to
///    interpolate the sub-pixel intensity maximum. The quadratic position is
///    used only when it agrees with the CoM (within 0.5 px); otherwise the CoM
///    is kept as-is.
///
/// When `max_elongation` is set in config, blobs with elongation ratio
/// (major/minor axis) exceeding the threshold are rejected as non-stellar.
/// The elongation test uses the **intensity-weighted** second moments —
/// the very same moments reported as `cov` (geometric moments admit a
/// slightly different set of marginal blobs — saturated stars with large
/// halos, etc. — which destabilizes downstream calibration on dense fields
/// like TESS).
///
/// This loop is ~2% of extraction wall-clock, so it is left sequential even
/// under the `parallel` feature — the threading overhead would not pay off and
/// keeps the two builds bit-identical here.
fn compute_blob_centroids(
    gray: &[f32],
    labels: &[u32],
    components: &[Component],
    width: u32,
    height: u32,
    config: &CentroidExtractionConfig,
) -> Vec<RawCentroid> {
    let w = width as usize;
    let h = height as usize;

    // Reused across blobs to avoid a fresh allocation per component (dense
    // fields can have thousands). The closure is `FnMut`, so it may borrow and
    // `clear()` this buffer each iteration.
    let mut annulus_vals: Vec<f32> = Vec::new();

    components
        .iter()
        .enumerate()
        .filter_map(|(idx, comp)| {
            let blob_label = (idx + 1) as u32;
            let pixel_count = comp.area as usize;
            if pixel_count < config.min_pixels || pixel_count > config.max_pixels {
                return None;
            }

            // Bounding box (numeris uses (row, col) with inclusive max).
            let min_row = comp.bbox_min.0 as usize;
            let max_row = comp.bbox_max.0 as usize;
            let min_col = comp.bbox_min.1 as usize;
            let max_col = comp.bbox_max.1 as usize;

            // Reference pixel = bbox top-left, to keep moments numerically stable.
            let ref_col = min_col;
            let ref_row = min_row;

            // --- Per-blob local background from annulus ---
            // Expand bounding box by margin, collect non-blob pixels
            const ANNULUS_MARGIN: usize = 5;
            let r0 = min_row.saturating_sub(ANNULUS_MARGIN);
            let r1 = (max_row + ANNULUS_MARGIN + 1).min(h);
            let c0 = min_col.saturating_sub(ANNULUS_MARGIN);
            let c1 = (max_col + ANNULUS_MARGIN + 1).min(w);

            annulus_vals.clear();
            for r in r0..r1 {
                let row_off = r * w;
                for c in c0..c1 {
                    let i = row_off + c;
                    if labels[i] == 0 {
                        annulus_vals.push(gray[i]);
                    }
                }
            }

            // Median of annulus (residual local background in bg-subtracted image).
            let local_bg = median_f32(&mut annulus_vals) as f64;

            // --- Single moment pass: intensity-weighted moments with the
            // annulus-local background, tracking the peak in the same sweep ---
            let mut sum_x = 0.0_f64;
            let mut sum_y = 0.0_f64;
            let mut sum_xx = 0.0_f64;
            let mut sum_yy = 0.0_f64;
            let mut sum_xy = 0.0_f64;
            let mut sum_i = 0.0_f64;
            let mut peak_val = f32::NEG_INFINITY;
            let mut peak_col: usize = ref_col;
            let mut peak_row: usize = ref_row;

            for r in min_row..=max_row {
                let row_off = r * w;
                for c in min_col..=max_col {
                    let i = row_off + c;
                    if labels[i] != blob_label {
                        continue;
                    }
                    let raw = gray[i];
                    if raw > peak_val {
                        peak_val = raw;
                        peak_col = c;
                        peak_row = r;
                    }
                    let intensity = (raw as f64 - local_bg).max(0.0);
                    let dx = c as f64 - ref_col as f64;
                    let dy = r as f64 - ref_row as f64;
                    sum_x += dx * intensity;
                    sum_y += dy * intensity;
                    sum_xx += dx * dx * intensity;
                    sum_yy += dy * dy * intensity;
                    sum_xy += dx * dy * intensity;
                    sum_i += intensity;
                }
            }

            if sum_i <= 0.0 {
                return None;
            }

            let dx_bar = sum_x / sum_i;
            let dy_bar = sum_y / sum_i;
            let xbar = ref_col as f64 + dx_bar;
            let ybar = ref_row as f64 + dy_bar;
            let cxx = sum_xx / sum_i - dx_bar * dx_bar;
            let cyy = sum_yy / sum_i - dy_bar * dy_bar;
            let cxy = sum_xy / sum_i - dx_bar * dy_bar;

            // Elongation filter — judged on the same intensity-weighted
            // moments reported as `cov` (they were historically computed
            // against the *global* background in a separate first pass,
            // so elongation and cov could disagree on marginal blobs).
            if let Some(max_elong) = config.max_elongation {
                if elongation_from_cov(cxx, cyy, cxy) > max_elong {
                    return None;
                }
            }

            let (pc, pr) = (peak_col, peak_row);
            // 3x3 grid of background-subtracted values around the peak
            let v = |dy: isize, dx: isize| -> f64 {
                let r = (pr as isize + dy) as usize;
                let c = (pc as isize + dx) as usize;
                gray[r * w + c] as f64 - local_bg
            };

            // --- Hot-pixel / cosmic-ray sharpness gate ---
            if let Some(max_sharp) = config.max_sharpness {
                if let Some(s) = peak_sharpness((pc, pr), (w, h), v) {
                    if s > max_sharp as f64 {
                        return None;
                    }
                }
            }

            // --- Quadratic peak refinement (shared gate; see accepted_peak_refine) ---
            // Skipped when the peak is saturated: a flat-topped or bloomed
            // profile has no meaningful sub-pixel maximum, and a 2-3 px flat
            // top can still "pass" the parabola with a skewed vertex.
            let mut final_x = xbar;
            let mut final_y = ybar;
            let saturated = config.saturation_level.is_some_and(|s| peak_val >= s);
            if !saturated {
                if let Some((qx, qy)) =
                    accepted_peak_refine(pixel_count, (pc, pr), (w, h), (xbar, ybar), v)
                {
                    final_x = qx;
                    final_y = qy;
                }
            }

            Some(RawCentroid {
                x_px: final_x as f32,
                y_px: final_y as f32,
                mass: sum_i as f32,
                cov: crate::Matrix2::new([[cxx as f32, cxy as f32], [cxy as f32, cyy as f32]]),
            })
        })
        .collect()
}
