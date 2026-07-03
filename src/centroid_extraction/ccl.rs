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
    accepted_peak_refine, median_f32, midpoint_f32, sort_and_truncate_by_mass,
    CentroidExtractionConfig, CentroidExtractionResult,
};
use crate::centroid::Centroid;
use crate::error::{Error, Result};

/// Parallelism dispatch for the centroid-extraction hot paths.
///
/// Each helper has two cfg-gated twins: a [Rayon](https://docs.rs/rayon)
/// work-stealing version under the `parallel` feature and a plain sequential
/// version otherwise. The feature flag lives only here, so the two paths cannot
/// drift apart and the call sites read identically in both configurations.
///
/// All helpers are deterministic: the element-wise maps write disjoint outputs
/// and `map_indices` / `for_each_chunk_mut` assign each index or chunk to a
/// fixed output slot, so results are independent of thread count and the
/// non-`parallel` build is bit-identical to the original sequential code.
///
/// Scope is deliberately narrow. Profiling (`smrecording.fits`, 2.1 Mpix) shows
/// `estimate_local_background` is ~60% of extraction wall-clock; the per-blob
/// centroid loop is ~2% and connected-component labeling lives in numeris and
/// is sequential there, so neither is parallelized here.
mod par {
    #[cfg(feature = "parallel")]
    use rayon::prelude::*;

    /// `(a - b).max(0.0)` element-wise (background subtraction with clamp).
    #[cfg(feature = "parallel")]
    pub fn map_subtract_clamp(a: &[f32], b: &[f32]) -> Vec<f32> {
        a.par_iter()
            .zip(b.par_iter())
            .map(|(&v, &bg)| (v - bg).max(0.0))
            .collect()
    }
    #[cfg(not(feature = "parallel"))]
    pub fn map_subtract_clamp(a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter()
            .zip(b.iter())
            .map(|(&v, &bg)| (v - bg).max(0.0))
            .collect()
    }

    /// `a - b` element-wise (background subtraction, unclamped).
    #[cfg(feature = "parallel")]
    pub fn map_subtract(a: &[f32], b: &[f32]) -> Vec<f32> {
        a.par_iter()
            .zip(b.par_iter())
            .map(|(&v, &bg)| v - bg)
            .collect()
    }
    #[cfg(not(feature = "parallel"))]
    pub fn map_subtract(a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b.iter()).map(|(&v, &bg)| v - bg).collect()
    }

    /// Map `f` over `0..n` into a `Vec`, preserving index order.
    #[cfg(feature = "parallel")]
    pub fn map_indices<T, F>(n: usize, f: F) -> Vec<T>
    where
        T: Send,
        F: Fn(usize) -> T + Sync + Send,
    {
        (0..n).into_par_iter().map(f).collect()
    }
    #[cfg(not(feature = "parallel"))]
    pub fn map_indices<T, F>(n: usize, f: F) -> Vec<T>
    where
        F: Fn(usize) -> T,
    {
        (0..n).map(f).collect()
    }

    /// Apply `f(i, chunk)` to each disjoint `chunk_len`-sized chunk of `buf`.
    /// `buf.len()` must be a multiple of `chunk_len` (one chunk per image row).
    #[cfg(feature = "parallel")]
    pub fn for_each_chunk_mut<T, F>(buf: &mut [T], chunk_len: usize, f: F)
    where
        T: Send,
        F: Fn(usize, &mut [T]) + Sync + Send,
    {
        buf.par_chunks_mut(chunk_len)
            .enumerate()
            .for_each(|(i, c)| f(i, c));
    }
    #[cfg(not(feature = "parallel"))]
    pub fn for_each_chunk_mut<T, F>(buf: &mut [T], chunk_len: usize, mut f: F)
    where
        F: FnMut(usize, &mut [T]),
    {
        for (i, c) in buf.chunks_mut(chunk_len).enumerate() {
            f(i, c);
        }
    }
}

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

    // ── Step 1: local background subtraction ──
    // If local_bg_block_size is set, estimate and subtract a spatially varying
    // background model. This is critical for images with nebulosity, Milky Way
    // emission, vignetting, or other large-scale intensity gradients. Without
    // it the input is used as-is (borrowed — no full-image copies).
    let gray: Cow<[f32]>;
    let local_bg: Option<Vec<f32>>;
    if let Some(block_size) = config.local_bg_block_size {
        let bg = estimate_local_background(gray_input, width, height, block_size);
        gray = Cow::Owned(par::map_subtract_clamp(gray_input, &bg));
        local_bg = Some(bg);
    } else {
        gray = Cow::Borrowed(gray_input);
        local_bg = None;
    }
    let gray: &[f32] = &gray;

    // ── Step 2: estimate residual background noise ──
    // Use unclamped residuals for noise estimation so the lower half of the
    // distribution is preserved (clamping to 0 destroys it).
    let noise_input: Cow<[f32]> = if let Some(ref bg) = local_bg {
        Cow::Owned(par::map_subtract(gray_input, bg))
    } else {
        Cow::Borrowed(gray_input)
    };
    let (bg_mean, bg_sigma) = estimate_background(&noise_input, width, height, config);
    let threshold = bg_mean + config.sigma_threshold * bg_sigma;

    // ── Step 3: optional matched filter for thresholding only ──
    // When `matched_filter_sigma` is set, the bg-subtracted residual is
    // convolved with a Gaussian and threshold/CCL run on the filtered copy.
    // Centroids are still measured on the unfiltered `gray`, so intensities
    // and CoM positions are unaffected. Under the `parallel` feature numeris's
    // gaussian_blur runs multi-threaded.
    let filtered: Option<Vec<f32>> = match config.matched_filter_sigma {
        Some(sigma) if sigma.is_finite() && sigma > 0.0 => {
            let mat = DynMatrix::<f32>::from_vec(w, h, gray.to_vec());
            Some(gaussian_blur(&mat, sigma, BorderMode::Replicate).into_vec())
        }
        _ => None,
    };
    let thresh_src: &[f32] = filtered.as_deref().unwrap_or(gray);

    // ── Step 4: threshold and label blobs ──
    // Build a u8 mask DynMatrix in proper (h, w) layout for CCL — its
    // bbox/labels conventions assume the supplied dimensions match the image.
    let mask = DynMatrix::<u8>::from_fn(h, w, |r, c| {
        if thresh_src[r * w + c] > threshold {
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
    // Use the local-background-subtracted image for centroid weighting so that
    // the intensity weights reflect only the stellar signal, not the gradient.
    let bg_for_centroids = if local_bg.is_some() {
        // Already subtracted — use 0 as the level
        0.0
    } else {
        bg_mean
    };
    let raw_centroids = compute_blob_centroids(
        gray,
        &labels,
        &components,
        width,
        height,
        bg_for_centroids,
        config,
    );
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
        threshold,
        num_blobs_raw,
    })
}

/// Estimate a spatially varying background by computing block medians and
/// interpolating between block centers.
///
/// The image is divided into `block_size × block_size` tiles. For each tile,
/// the median pixel value is computed (ignoring zeros). A smooth background
/// surface is then reconstructed via bilinear interpolation between tile
/// centers.
///
/// This effectively removes large-scale structure (nebulosity, Milky Way
/// emission, vignetting) while preserving point sources (stars).
///
/// This is the dominant extraction stage (~60% of wall-clock); under the
/// `parallel` feature the per-block medians and the per-row interpolation both
/// fan out across threads. Each block / row writes its own output slot, so the
/// result is identical to the sequential path.
fn estimate_local_background(pixels: &[f32], width: u32, height: u32, block_size: u32) -> Vec<f32> {
    let w = width as usize;
    let h = height as usize;
    let bs = block_size as usize;

    // Number of blocks in each dimension
    let nx = w.div_ceil(bs);
    let ny = h.div_ceil(bs);

    // Compute median for each block. Blocks are independent and each writes its
    // own index, so this maps in parallel under the `parallel` feature.
    let block_medians: Vec<f32> = par::map_indices(nx * ny, |bi| {
        let bx = bi % nx;
        let by = bi / nx;
        let x0 = bx * bs;
        let y0 = by * bs;
        let x1 = (x0 + bs).min(w);
        let y1 = (y0 + bs).min(h);

        let mut vals: Vec<f32> = Vec::with_capacity(bs * bs);
        for y in y0..y1 {
            for x in x0..x1 {
                let v = pixels[y * w + x];
                if v > 0.0 && v.is_finite() {
                    vals.push(v);
                }
            }
        }

        midpoint_f32(&mut vals)
    });

    // Bilinearly interpolate between block centers to produce a smooth
    // background estimate at every pixel. Each row depends only on the shared,
    // immutable block medians, so rows are filled in parallel over disjoint
    // output slices.
    let mut background = vec![0.0f32; w * h];
    let half_bs = bs as f32 / 2.0;

    par::for_each_chunk_mut(&mut background, w, |y, row| {
        // Position in block-center coordinates (row component)
        let by_f = (y as f32 - half_bs) / bs as f32;
        let by0 = (by_f.floor() as isize).max(0).min(ny as isize - 1) as usize;
        let by1 = (by0 + 1).min(ny - 1);
        let fy = (by_f - by0 as f32).clamp(0.0, 1.0);

        for (x, px) in row.iter_mut().enumerate() {
            let bx_f = (x as f32 - half_bs) / bs as f32;
            let bx0 = (bx_f.floor() as isize).max(0).min(nx as isize - 1) as usize;
            let bx1 = (bx0 + 1).min(nx - 1);
            let fx = (bx_f - bx0 as f32).clamp(0.0, 1.0);

            let m00 = block_medians[by0 * nx + bx0];
            let m10 = block_medians[by0 * nx + bx1];
            let m01 = block_medians[by1 * nx + bx0];
            let m11 = block_medians[by1 * nx + bx1];

            *px = m00 * (1.0 - fx) * (1.0 - fy)
                + m10 * fx * (1.0 - fy)
                + m01 * (1.0 - fx) * fy
                + m11 * fx * fy;
        }
    });

    background
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
/// The elongation test uses **intensity-weighted** second moments (with the
/// global background subtracted), matching the original behavior — geometric
/// moments admit a slightly different set of marginal blobs (saturated stars
/// with large halos, etc.), which destabilizes downstream calibration on
/// dense fields like TESS.
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
    bg_level: f32,
    config: &CentroidExtractionConfig,
) -> Vec<RawCentroid> {
    let w = width as usize;
    let h = height as usize;
    let bg_level_f64 = bg_level as f64;

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

            // --- Pass 1: intensity-weighted moments with global bg + peak ---
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
                    let intensity = (raw as f64 - bg_level_f64).max(0.0);
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

            // Elongation filter on intensity-weighted moments
            if let Some(max_elong) = config.max_elongation {
                let dx_bar = sum_x / sum_i;
                let dy_bar = sum_y / sum_i;
                let cxx = sum_xx / sum_i - dx_bar * dx_bar;
                let cyy = sum_yy / sum_i - dy_bar * dy_bar;
                let cxy = sum_xy / sum_i - dx_bar * dy_bar;
                let trace = cxx + cyy;
                let det = cxx * cyy - cxy * cxy;
                let disc = (trace * trace - 4.0 * det).max(0.0).sqrt();
                let lambda_max = (trace + disc) / 2.0;
                let lambda_min = (trace - disc).max(1e-12) / 2.0;
                let elongation = (lambda_max / lambda_min).sqrt() as f32;
                if elongation > max_elong {
                    return None;
                }
            }

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

            // --- Pass 2: re-accumulate intensity-weighted moments with local bg ---
            sum_x = 0.0;
            sum_y = 0.0;
            sum_xx = 0.0;
            sum_yy = 0.0;
            sum_xy = 0.0;
            sum_i = 0.0;

            for r in min_row..=max_row {
                let row_off = r * w;
                for c in min_col..=max_col {
                    let i = row_off + c;
                    if labels[i] != blob_label {
                        continue;
                    }
                    let intensity = (gray[i] as f64 - local_bg).max(0.0);
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

            // --- Quadratic peak refinement (shared gate; see accepted_peak_refine) ---
            let mut final_x = xbar;
            let mut final_y = ybar;

            let (pc, pr) = (peak_col, peak_row);
            // 3x3 grid of background-subtracted values around the peak
            let v = |dy: isize, dx: isize| -> f64 {
                let r = (pr as isize + dy) as usize;
                let c = (pc as isize + dx) as usize;
                gray[r * w + c] as f64 - local_bg
            };
            if let Some((qx, qy)) =
                accepted_peak_refine(pixel_count, (pc, pr), (w, h), (xbar, ybar), v)
            {
                final_x = qx;
                final_y = qy;
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
