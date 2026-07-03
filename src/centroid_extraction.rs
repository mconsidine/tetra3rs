//! Extract star centroids from an astronomical image.
//!
//! This module provides functions to detect and locate stars in pixel data by:
//! 1. Converting the image to grayscale floating-point values
//! 2. Estimating and subtracting the background (sigma-clipped median)
//! 3. Thresholding to identify bright pixels
//! 4. Labeling connected components (blobs)
//! 5. Computing intensity-weighted centroids for each blob, with:
//!    - Per-blob local background from an annulus of non-blob pixels
//!    - Quadratic peak refinement (2D fit to 3×3 around peak pixel)
//!
//! Requires the `image` feature to be enabled.
//!
//! Entry points:
//! - [`extract_centroids_from_image`] for an already-decoded
//!   [`image::DynamicImage`]. The caller is responsible for decoding the
//!   file (using whichever `image` feature flags suit their needs).
//! - [`extract_centroids_from_raw`] for raw grayscale `f32` pixel data —
//!   useful for FITS, camera SDK output, or any other non-standard source.
//! - [`extract_centroids_fast`] is a single-pass "adequate star tracker"
//!   alternative: it reads each pixel once (coarse-grid background +
//!   run-length connected-component moments) for markedly lower latency, at
//!   the cost of faint-star sensitivity and sub-pixel accuracy. The two
//!   functions above stay the default and the right choice for calibration.
//!
//! With the `parallel` feature, the dominant local-background stage and the
//! full-image element-wise maps of the connected-component path run
//! multi-threaded via rayon; results are bit-identical to the sequential
//! build. (The fast single-pass path is sequential.)
//!
//! # Example
//!
//! ```no_run
//! use tetra3::centroid_extraction::{CentroidExtractionConfig, extract_centroids_from_image};
//!
//! let img = image::open("my_star_image.png").unwrap();
//! let config = CentroidExtractionConfig::default();
//! let result = extract_centroids_from_image(&img, &config).unwrap();
//! println!("Found {} stars", result.centroids.len());
//! ```

use crate::centroid::Centroid;
use crate::error::{Error, Result};
use image::GenericImageView;
use numeris::imageproc::{
    connected_components_with_label_buffer, gaussian_blur, BorderMode, Component, Connectivity,
};
use numeris::DynMatrix;

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

/// Configuration for centroid extraction from an image.
#[derive(Debug, Clone)]
pub struct CentroidExtractionConfig {
    /// Number of sigma above background to use as the detection threshold.
    /// Stars brighter than `background + sigma_threshold * noise` are detected.
    /// Default: 5.0
    pub sigma_threshold: f32,

    /// Minimum number of pixels in a blob to be considered a star.
    /// Helps filter out hot pixels and noise.
    /// Default: 3
    pub min_pixels: usize,

    /// Maximum number of pixels in a blob to be considered a star.
    /// Helps filter out very large extended objects.
    /// Set high enough to include saturated bright stars with large halos.
    /// Default: 10000
    pub max_pixels: usize,

    /// Maximum number of centroids to return, sorted by brightness (mass).
    /// If `None`, all detected centroids are returned.
    /// Default: None
    pub max_centroids: Option<usize>,

    /// Number of iterations for sigma-clipped background estimation.
    /// Default: 5
    pub sigma_clip_iterations: usize,

    /// Sigma clipping factor for background estimation.
    /// Pixels more than this many sigma from the mean are excluded.
    /// Default: 3.0
    pub sigma_clip_factor: f32,

    /// Whether to use 8-connectivity (true) or 4-connectivity (false) for
    /// connected component labeling.
    /// Default: true (8-connectivity)
    pub use_8_connectivity: bool,

    /// Block size (in pixels) for local background estimation.
    ///
    /// When set to `Some(n)`, the image is divided into `n×n` blocks and
    /// the median value in each block is computed. A smooth background
    /// model is created by bilinear interpolation between block centers
    /// and subtracted before star detection. This removes large-scale
    /// gradients from nebulosity, Milky Way emission, or vignetting.
    ///
    /// A good starting value is 32-128 pixels, or roughly 1-3% of the
    /// image width. Smaller blocks follow finer structure but risk
    /// subtracting real stars.
    ///
    /// When `None`, only global background subtraction is used (original
    /// behavior).
    ///
    /// Default: Some(64)
    pub local_bg_block_size: Option<u32>,

    /// Maximum allowed elongation ratio (major/minor axis) for a detected
    /// blob. Blobs more elongated than this are rejected as non-stellar
    /// (e.g. cosmic rays, satellite trails, diffraction spikes).
    ///
    /// A value of 2.0 means the blob can be at most 2× longer than wide.
    /// Set to a large value (e.g. 100) or `None` to disable.
    ///
    /// Default: None (disabled)
    pub max_elongation: Option<f32>,

    /// Apply a Gaussian matched filter to the bg-subtracted image before
    /// thresholding. When `Some(sigma)`, the image is convolved with a
    /// separable 1-D Gaussian (σ in pixels, kernel truncated at 3σ). The
    /// filtered image is used **only** to form the detection mask —
    /// centroid positions and intensities are still measured on the
    /// unfiltered bg-subtracted image, so photometry is unaffected.
    ///
    /// A matched filter boosts point-source SNR by concentrating the
    /// star's PSF into fewer effective pixels before thresholding. The
    /// gain is largest for faint stars in noisy or dense images. The
    /// filter has a broad optimum: σ within a factor of ~2 of the true
    /// PSF width still recovers nearly all the SNR.
    ///
    /// When enabled, consider lowering `sigma_threshold` (e.g. to 2.5–3.0)
    /// since the filtered image has lower noise.
    ///
    /// Default: None (disabled).
    pub matched_filter_sigma: Option<f32>,
}

impl Default for CentroidExtractionConfig {
    fn default() -> Self {
        Self {
            sigma_threshold: 5.0,
            min_pixels: 3,
            max_pixels: 10000,
            max_centroids: None,
            sigma_clip_iterations: 5,
            sigma_clip_factor: 3.0,
            use_8_connectivity: true,
            local_bg_block_size: Some(64),
            max_elongation: Some(3.0),
            matched_filter_sigma: None,
        }
    }
}

/// Result of centroid extraction, containing the centroids and diagnostic info.
#[derive(Debug, Clone)]
pub struct CentroidExtractionResult {
    /// Extracted centroids in pixel coordinates, with (0, 0) at the image center.
    /// +X is right (increasing column), +Y is down (increasing row).
    pub centroids: Vec<Centroid>,

    /// Image width in pixels.
    pub image_width: u32,

    /// Image height in pixels.
    pub image_height: u32,

    /// Estimated background level (in image intensity units).
    pub background_mean: f32,

    /// Estimated background noise standard deviation.
    pub background_sigma: f32,

    /// Detection threshold used (background_mean + sigma_threshold * background_sigma).
    pub threshold: f32,

    /// Number of blobs found before size filtering.
    pub num_blobs_raw: usize,
}

/// Extract star centroids from an already-decoded [`image::DynamicImage`].
///
/// Performs background subtraction, blob detection, and centroid computation
/// on an in-memory image. Centroids are returned in pixel coordinates with the
/// origin at the image center, suitable for use with
/// [`SolverDatabase::solve_from_centroids`].
///
/// To load from a file, decode it with `image::open(path)?` (which requires
/// the appropriate `image` crate format features in your own `Cargo.toml`)
/// and pass the resulting `DynamicImage` here.
pub fn extract_centroids_from_image(
    img: &image::DynamicImage,
    config: &CentroidExtractionConfig,
) -> Result<CentroidExtractionResult> {
    let (width, height) = img.dimensions();
    let gray = to_grayscale_f32(img);
    extract_from_gray(&gray, width, height, config)
}

/// Extract star centroids from raw grayscale pixel data.
///
/// This is useful when you have pixel data that isn't in a standard image format,
/// e.g. from a camera SDK or FITS file parsed externally.
///
/// # Arguments
///
/// * `pixels` - Row-major grayscale pixel values (length must equal `width * height`)
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `config` - Extraction configuration parameters
pub fn extract_centroids_from_raw(
    pixels: &[f32],
    width: u32,
    height: u32,
    config: &CentroidExtractionConfig,
) -> Result<CentroidExtractionResult> {
    let expected = (width as usize) * (height as usize);
    if pixels.len() != expected {
        return Err(Error::InvalidInput(format!(
            "Pixel data length ({}) does not match width*height ({}x{}={})",
            pixels.len(),
            width,
            height,
            expected,
        )));
    }
    extract_from_gray(pixels, width, height, config)
}

// ─── Fast single-pass star-tracker path ─────────────────────────────────────

/// Configuration for [`extract_centroids_fast`].
///
/// Deliberately small — the four knobs a single-pass detector needs. None of
/// the connected-component path's quality filters (block-interpolated
/// background, elongation, per-blob annulus background, sub-pixel agreement
/// gates) appear, because this path trades them away for speed.
#[derive(Debug, Clone)]
pub struct FastCentroidConfig {
    /// Detection threshold in noise sigmas above the local background. A pixel
    /// is "lit" when `value > bg(x, y) + sigma_threshold · σ`. Default: 5.0
    pub sigma_threshold: f32,

    /// Coarse background-grid block size in pixels. The background is estimated
    /// once on a `bg_grid`-spaced grid (from a subsampled pre-pass, ~1/64 the
    /// pixels) and bilinearly interpolated during the main sweep, so gradients
    /// (vignetting, Milky Way) are handled without a full-image background
    /// stage. Larger blocks are cheaper but follow gradients more coarsely.
    /// Default: 64
    pub bg_grid: u32,

    /// Minimum pixels in a lit region to count as a star — rejects single hot
    /// pixels and cosmic-ray specks. Default: 2
    pub min_pixels: usize,

    /// Maximum number of centroids to return, brightest first. `None` returns
    /// all detections. For plate solving / tracking a few dozen is plenty.
    /// Default: None
    pub max_centroids: Option<usize>,
}

impl Default for FastCentroidConfig {
    fn default() -> Self {
        Self {
            sigma_threshold: 5.0,
            bg_grid: 64,
            min_pixels: 2,
            max_centroids: None,
        }
    }
}

/// Per-region moment accumulator for the single-pass detector. One per pixel
/// **run**; runs that connect across rows are merged by union-find at the end.
#[derive(Clone, Copy)]
struct Region {
    parent: u32,
    sum_w: f64,  // Σ (value − bg), the background-subtracted flux
    sum_wx: f64, // Σ x·(value − bg)
    sum_wy: f64, // Σ y·(value − bg)
    npix: u32,
    peak_val: f32,
    peak_x: u32,
    peak_y: u32,
}

/// Fast single-pass centroid extraction — the "adequate star tracker" path.
///
/// An alternative to [`extract_centroids_from_raw`] that reads each pixel
/// **once**: a cheap subsampled pre-pass builds a coarse background grid, then
/// a single raster sweep thresholds against the bilinearly-interpolated
/// background, groups lit pixels into connected regions via run-length +
/// union-find (accumulating intensity-weighted moments inline), and emits a
/// center-of-mass per region. No convolution, no full-image background buffer,
/// no second pass — so it is memory-bandwidth-bound rather than compute-bound,
/// and markedly faster than the connected-component path (which stays the
/// default and the right choice for calibration / faint-star work).
///
/// # Trade-offs
///
/// - No matched filter, so faint-star sensitivity is lower — sized for the
///   brightest stars a tracker locks onto, not deep detection.
/// - Center-of-mass is threshold-clipped: sub-pixel accuracy is ~0.1 px for
///   bright stars, degrading for faint ones. A 3×3 parabola refine on the peak
///   ([`quadratic_peak_offset`]) sharpens it when the region is large enough
///   and the fit agrees with the CoM, matching the CCL path's gate.
/// - A global noise σ is used (adequate when read/shot noise is roughly
///   uniform even where the background level is not).
///
/// Returns the same [`CentroidExtractionResult`] as the CCL path (centroids in
/// image-center-origin coordinates, brightest first), so it is a drop-in for
/// [`SolverDatabase::solve_from_centroids`](crate::SolverDatabase::solve_from_centroids).
/// `background_mean` is the median of the coarse grid, `background_sigma` the
/// global noise σ, `threshold` a representative `bg_mean + k·σ`, and
/// `num_blobs_raw` the region count before the `min_pixels` filter.
pub fn extract_centroids_fast(
    pixels: &[f32],
    width: u32,
    height: u32,
    config: &FastCentroidConfig,
) -> Result<CentroidExtractionResult> {
    let w = width as usize;
    let h = height as usize;
    let expected = w * h;
    if pixels.len() != expected {
        return Err(Error::InvalidInput(format!(
            "Pixel data length ({}) does not match width*height ({}x{}={})",
            pixels.len(),
            width,
            height,
            expected,
        )));
    }
    if !(config.sigma_threshold.is_finite() && config.sigma_threshold > 0.0) {
        return Err(Error::InvalidInput(format!(
            "sigma_threshold must be finite and positive, got {}",
            config.sigma_threshold
        )));
    }
    if config.bg_grid == 0 {
        return Err(Error::InvalidInput("bg_grid must be >= 1".into()));
    }
    if w < 2 || h < 2 {
        return Err(Error::InvalidInput("image must be at least 2x2".into()));
    }

    // ── Pre-pass: coarse background grid + global noise σ (subsampled) ──
    let block = config.bg_grid as usize;
    let (bg_grid, nx, ny, sigma) = coarse_background(pixels, w, h, block);
    let k = config.sigma_threshold;

    // ── Single raster sweep: run-length detection + union-find moments ──
    let mut regions: Vec<Region> = Vec::new();
    // Runs of the previous / current row: (col_start, col_end_inclusive, label).
    let mut prev: Vec<(u32, u32, u32)> = Vec::new();
    let mut cur: Vec<(u32, u32, u32)> = Vec::new();

    for r in 0..h {
        cur.clear();
        let row = r * w;
        let mut active: Option<(u32, Region)> = None; // (start_col, accumulator)

        for c in 0..w {
            let bg = bilinear_grid(&bg_grid, nx, ny, block, c, r);
            let p = pixels[row + c];
            let lit = p.is_finite() && p > bg + k * sigma;

            if lit {
                let weight = (p - bg).max(0.0) as f64;
                // Start a new run only when one isn't already open (building the
                // Region eagerly every lit pixel would be wasteful).
                if active.is_none() {
                    active = Some((
                        c as u32,
                        Region {
                            parent: 0, // set when finalized
                            sum_w: 0.0,
                            sum_wx: 0.0,
                            sum_wy: 0.0,
                            npix: 0,
                            peak_val: f32::NEG_INFINITY,
                            peak_x: c as u32,
                            peak_y: r as u32,
                        },
                    ));
                }
                let reg = &mut active.as_mut().unwrap().1;
                reg.sum_w += weight;
                reg.sum_wx += weight * c as f64;
                reg.sum_wy += weight * r as f64;
                reg.npix += 1;
                if p > reg.peak_val {
                    reg.peak_val = p;
                    reg.peak_x = c as u32;
                    reg.peak_y = r as u32;
                }
            } else if let Some((start, mut reg)) = active.take() {
                let label = regions.len() as u32;
                reg.parent = label;
                regions.push(reg);
                cur.push((start, c as u32 - 1, label));
            }
        }
        if let Some((start, mut reg)) = active.take() {
            let label = regions.len() as u32;
            reg.parent = label;
            regions.push(reg);
            cur.push((start, w as u32 - 1, label));
        }

        // Merge current-row runs with 8-connected previous-row runs. Both lists
        // are column-sorted, so a two-pointer sweep finds all overlaps.
        let (mut i, mut j) = (0usize, 0usize);
        while i < cur.len() && j < prev.len() {
            let (cs, ce, cl) = cur[i];
            let (ps, pe, pl) = prev[j];
            if ce + 1 < ps {
                i += 1; // current run ends left of (and not adjacent to) prev
            } else if pe + 1 < cs {
                j += 1; // prev run ends left of current
            } else {
                union(&mut regions, cl, pl);
                if ce < pe {
                    i += 1;
                } else {
                    j += 1;
                }
            }
        }

        std::mem::swap(&mut prev, &mut cur);
    }

    // ── Resolve union-find: fold each region into its root ──
    let n_labels = regions.len();
    for lab in 0..n_labels {
        let root = find(&mut regions, lab as u32) as usize;
        if root != lab {
            let (sw, swx, swy, np, pv, px, py) = {
                let c = &regions[lab];
                (
                    c.sum_w, c.sum_wx, c.sum_wy, c.npix, c.peak_val, c.peak_x, c.peak_y,
                )
            };
            let rt = &mut regions[root];
            rt.sum_w += sw;
            rt.sum_wx += swx;
            rt.sum_wy += swy;
            rt.npix += np;
            if pv > rt.peak_val {
                rt.peak_val = pv;
                rt.peak_x = px;
                rt.peak_y = py;
            }
        }
    }

    // ── Emit one centroid per root region ──
    // Origin at the geometric image center (W-1)/2, (H-1)/2 (see the CCL path).
    let cx = (width - 1) as f32 / 2.0;
    let cy = (height - 1) as f32 / 2.0;
    let mut centroids: Vec<Centroid> = Vec::new();
    let mut num_blobs_raw = 0usize;
    for lab in 0..n_labels {
        if find(&mut regions, lab as u32) as usize != lab {
            continue; // not a root
        }
        num_blobs_raw += 1;
        let reg = regions[lab];
        if (reg.npix as usize) < config.min_pixels || reg.sum_w <= 0.0 {
            continue;
        }
        let mut fx = reg.sum_wx / reg.sum_w;
        let mut fy = reg.sum_wy / reg.sum_w;

        // Optional 3×3 parabola refine on the raw image at the peak, gated to
        // agree with the CoM within 0.5 px (mirrors the CCL path).
        let (pc, pr) = (reg.peak_x as usize, reg.peak_y as usize);
        if reg.npix >= 5 && pc >= 1 && pr >= 1 && pc + 1 < w && pr + 1 < h {
            let bg = bilinear_grid(&bg_grid, nx, ny, block, pc, pr) as f64;
            let v = |dy: isize, dx: isize| -> f64 {
                let rr = (pr as isize + dy) as usize;
                let cc = (pc as isize + dx) as usize;
                pixels[rr * w + cc] as f64 - bg
            };
            if let Some((x_off, y_off)) = quadratic_peak_offset(v) {
                let (qx, qy) = (pc as f64 + x_off, pr as f64 + y_off);
                if (qx - fx).powi(2) + (qy - fy).powi(2) < 0.25 {
                    fx = qx;
                    fy = qy;
                }
            }
        }

        centroids.push(Centroid {
            x: fx as f32 - cx,
            y: fy as f32 - cy,
            mass: Some(reg.sum_w as f32),
            cov: None,
        });
    }

    centroids.sort_by(|a, b| {
        b.mass
            .unwrap_or(0.0)
            .partial_cmp(&a.mass.unwrap_or(0.0))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if let Some(max) = config.max_centroids {
        centroids.truncate(max);
    }

    let bg_mean = {
        let mut g = bg_grid.clone();
        let m = g.len() / 2;
        let (_, nth, _) = g.select_nth_unstable_by(m, |a, b| a.total_cmp(b));
        *nth
    };

    Ok(CentroidExtractionResult {
        centroids,
        image_width: width,
        image_height: height,
        background_mean: bg_mean,
        background_sigma: sigma,
        threshold: bg_mean + k * sigma,
        num_blobs_raw,
    })
}

/// Union-find root with path halving over the region accumulators.
fn find(regions: &mut [Region], mut x: u32) -> u32 {
    while regions[x as usize].parent != x {
        let parent = regions[x as usize].parent;
        regions[x as usize].parent = regions[parent as usize].parent; // halve
        x = regions[x as usize].parent;
    }
    x
}

/// Link the roots of two region labels.
fn union(regions: &mut [Region], a: u32, b: u32) {
    let ra = find(regions, a);
    let rb = find(regions, b);
    if ra != rb {
        regions[ra as usize].parent = rb;
    }
}

/// Coarse background grid + global noise σ from a subsampled pre-pass.
///
/// The image is divided into `block × block` cells; each cell's median is taken
/// from a strided subsample (~1/64 of the pixels for the default block), giving
/// an `nx × ny` background grid. The global noise σ is the RMS of the
/// below-median residuals across the subsample (the half-normal estimate,
/// robust to stars which only push the distribution upward).
///
/// Returns `(grid, nx, ny, sigma)` with `grid` row-major `nx`-wide.
fn coarse_background(
    pixels: &[f32],
    w: usize,
    h: usize,
    block: usize,
) -> (Vec<f32>, usize, usize, f32) {
    let nx = w.div_ceil(block);
    let ny = h.div_ceil(block);
    let stride = (block / 8).max(1);
    let mut grid = vec![0.0_f32; nx * ny];
    let mut sq_sum = 0.0_f64;
    let mut sq_n = 0usize;
    let mut samples: Vec<f32> = Vec::with_capacity((block / stride + 1).pow(2));

    for by in 0..ny {
        let y0 = by * block;
        let y1 = (y0 + block).min(h);
        for bx in 0..nx {
            let x0 = bx * block;
            let x1 = (x0 + block).min(w);
            samples.clear();
            let mut y = y0;
            while y < y1 {
                let row = y * w;
                let mut x = x0;
                while x < x1 {
                    let v = pixels[row + x];
                    if v.is_finite() {
                        samples.push(v);
                    }
                    x += stride;
                }
                y += stride;
            }
            let median = if samples.is_empty() {
                0.0
            } else {
                let m = samples.len() / 2;
                let (_, nth, _) = samples.select_nth_unstable_by(m, |a, b| a.total_cmp(b));
                *nth
            };
            grid[by * nx + bx] = median;
            // Below-median residuals → robust noise (uncontaminated by stars).
            for &v in samples.iter() {
                if v <= median {
                    let d = (v - median) as f64;
                    sq_sum += d * d;
                    sq_n += 1;
                }
            }
        }
    }

    let sigma = if sq_n > 0 {
        (sq_sum / sq_n as f64).sqrt() as f32
    } else {
        0.0
    };
    (grid, nx, ny, sigma)
}

/// Bilinear interpolation of a coarse background grid at pixel `(x, y)`.
///
/// Grid samples sit at block centers (`block/2 + bx·block`); positions outside
/// the centers clamp to the nearest edge cell.
fn bilinear_grid(grid: &[f32], nx: usize, ny: usize, block: usize, x: usize, y: usize) -> f32 {
    let half = block as f32 / 2.0;
    let fx = (x as f32 - half) / block as f32;
    let fy = (y as f32 - half) / block as f32;
    let bx0 = (fx.floor().max(0.0) as usize).min(nx - 1);
    let by0 = (fy.floor().max(0.0) as usize).min(ny - 1);
    let bx1 = (bx0 + 1).min(nx - 1);
    let by1 = (by0 + 1).min(ny - 1);
    let tx = (fx - bx0 as f32).clamp(0.0, 1.0);
    let ty = (fy - by0 as f32).clamp(0.0, 1.0);
    let g = |bx: usize, by: usize| grid[by * nx + bx];
    let top = g(bx0, by0) * (1.0 - tx) + g(bx1, by0) * tx;
    let bot = g(bx0, by1) * (1.0 - tx) + g(bx1, by1) * tx;
    top * (1.0 - ty) + bot * ty
}

// ─── Internal helpers ──────────────────────────────────────────────────────

/// Shared extraction pipeline for both image and raw-pixel entry points.
fn extract_from_gray(
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
    // emission, vignetting, or other large-scale intensity gradients.
    let gray: Vec<f32>;
    let local_bg: Option<Vec<f32>>;
    if let Some(block_size) = config.local_bg_block_size {
        let bg = estimate_local_background(gray_input, width, height, block_size);
        gray = par::map_subtract_clamp(gray_input, &bg);
        local_bg = Some(bg);
    } else {
        gray = gray_input.to_vec();
        local_bg = None;
    }

    // ── Step 2: estimate residual background noise ──
    // Use unclamped residuals for noise estimation so the lower half of the
    // distribution is preserved (clamping to 0 destroys it).
    let noise_input = if let Some(ref bg) = local_bg {
        par::map_subtract(gray_input, bg)
    } else {
        gray_input.to_vec()
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
            let mat = DynMatrix::<f32>::from_vec(w, h, gray.clone());
            Some(gaussian_blur(&mat, sigma, BorderMode::Replicate).into_vec())
        }
        _ => None,
    };
    let thresh_src: &[f32] = filtered.as_deref().unwrap_or(&gray);

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
        &gray,
        &labels,
        &components,
        width,
        height,
        bg_for_centroids,
        config,
    );
    let num_blobs_raw = raw_centroids.len();

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

    // Sort by brightness (descending)
    centroids.sort_by(|a, b| {
        b.mass
            .unwrap_or(0.0)
            .partial_cmp(&a.mass.unwrap_or(0.0))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    if let Some(max) = config.max_centroids {
        centroids.truncate(max);
    }

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

        if vals.is_empty() {
            0.0
        } else {
            vals.sort_unstable_by(|a, b| a.total_cmp(b));
            vals[vals.len() / 2]
        }
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

/// Convert a DynamicImage to a Vec<f32> of grayscale values.
fn to_grayscale_f32(img: &image::DynamicImage) -> Vec<f32> {
    use image::DynamicImage;
    match img {
        // 16-bit images: normalize to [0, 65535] range as f32
        DynamicImage::ImageLuma16(g) => g.as_raw().iter().map(|&v| v as f32).collect(),
        DynamicImage::ImageLumaA16(g) => g.pixels().map(|p| p.0[0] as f32).collect(),
        DynamicImage::ImageRgb16(rgb) => rgb
            .pixels()
            .map(|p| {
                let [r, g, b] = p.0;
                0.2126 * r as f32 + 0.7152 * g as f32 + 0.0722 * b as f32
            })
            .collect(),
        DynamicImage::ImageRgba16(rgba) => rgba
            .pixels()
            .map(|p| {
                let [r, g, b, _] = p.0;
                0.2126 * r as f32 + 0.7152 * g as f32 + 0.0722 * b as f32
            })
            .collect(),
        // For 32-bit float images
        DynamicImage::ImageRgb32F(rgb) => rgb
            .pixels()
            .map(|p| {
                let [r, g, b] = p.0;
                0.2126 * r + 0.7152 * g + 0.0722 * b
            })
            .collect(),
        DynamicImage::ImageRgba32F(rgba) => rgba
            .pixels()
            .map(|p| {
                let [r, g, b, _] = p.0;
                0.2126 * r + 0.7152 * g + 0.0722 * b
            })
            .collect(),
        // 8-bit and other formats: convert via luma8
        _ => {
            let gray = img.to_luma8();
            gray.as_raw().iter().map(|&v| v as f32).collect()
        }
    }
}

/// Estimate background level and noise.
///
/// Uses the median as the background level and estimates noise from the
/// lower half of the pixel distribution (below the median). This is robust
/// to contamination from stars and nebulosity, which only bias upward.
///
/// The noise estimate uses sigma-clipping on the below-median pixels to
/// further reject any remaining outliers, then mirrors the lower-half RMS
/// to get the full Gaussian sigma.
fn estimate_background(
    gray: &[f32],
    _width: u32,
    _height: u32,
    config: &CentroidExtractionConfig,
) -> (f32, f32) {
    let mut values: Vec<f32> = gray.iter().copied().filter(|v| v.is_finite()).collect();
    if values.is_empty() {
        return (0.0, 0.0);
    }

    let n = values.len();

    // Median as robust background level. Only the median (an order statistic) is
    // needed, so partition in O(n) via select_nth instead of a full O(n log n)
    // sort. For even n the median averages the two central order statistics:
    // values[n/2] (the selected element) and values[n/2 - 1] (the max of the
    // lower partition that select_nth leaves to its left).
    let cmp = |a: &f32, b: &f32| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal);
    let median = {
        let (lower, nth, _) = values.select_nth_unstable_by(n / 2, cmp);
        if n.is_multiple_of(2) {
            let prev = lower.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            (prev + *nth) / 2.0
        } else {
            *nth
        }
    };

    // Estimate noise from pixels at or below the median (uncontaminated by stars).
    // These represent the "dark side" of the noise distribution.
    let mut low_half: Vec<f32> = values.iter().copied().filter(|&v| v <= median).collect();

    // Sigma-clip the lower half to reject any remaining outliers
    let mut sigma = 0.0_f32;
    for _ in 0..config.sigma_clip_iterations {
        if low_half.is_empty() {
            break;
        }
        let sum: f64 = low_half.iter().map(|&v| v as f64).sum();
        let mean_low = (sum / low_half.len() as f64) as f32;
        let var_sum: f64 = low_half
            .iter()
            .map(|&v| ((v - mean_low) as f64).powi(2))
            .sum();
        sigma = (var_sum / low_half.len() as f64).sqrt() as f32;
        if sigma < 1e-10 {
            break;
        }
        let lo = mean_low - config.sigma_clip_factor * sigma;
        let hi = mean_low + config.sigma_clip_factor * sigma;
        let before = low_half.len();
        low_half.retain(|&v| v >= lo && v <= hi);
        if low_half.len() == before {
            break; // converged
        }
    }

    (median, sigma)
}

/// Sub-pixel peak offset from a 2-D quadratic fit to a 3×3 neighborhood.
///
/// `v(dy, dx)` samples the (background-subtracted) surface at the peak pixel
/// plus integer offset `(dy, dx)`, `dx`/`dy` ∈ {−1, 0, 1}. Fits a bivariate
/// quadratic and returns the vertex offset `(x_off, y_off)` from the peak
/// pixel, or `None` when the fit is degenerate (near-flat Hessian) or
/// extrapolates beyond half a pixel (an unreliable peak — the caller should
/// fall back to the integer peak / center-of-mass). Shared by the
/// connected-component path ([`compute_blob_centroids`]) and the fast
/// DoG path ([`extract_centroids_fast`]).
fn quadratic_peak_offset(v: impl Fn(isize, isize) -> f64) -> Option<(f64, f64)> {
    let b = (v(0, 1) - v(0, -1)) / 2.0;
    let c_coeff = (v(1, 0) - v(-1, 0)) / 2.0;
    let d = (v(0, 1) + v(0, -1) - 2.0 * v(0, 0)) / 2.0;
    let f = (v(1, 0) + v(-1, 0) - 2.0 * v(0, 0)) / 2.0;
    let e = (v(1, 1) - v(1, -1) - v(-1, 1) + v(-1, -1)) / 4.0;

    let denom = 4.0 * d * f - e * e;
    if denom.abs() <= 1e-10 {
        return None;
    }
    let x_off = (e * c_coeff - 2.0 * f * b) / denom;
    let y_off = (e * b - 2.0 * d * c_coeff) / denom;
    if x_off.abs() <= 0.5 && y_off.abs() <= 0.5 {
        Some((x_off, y_off))
    } else {
        None
    }
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
            // select_nth partitions in O(n); for even length the lower central
            // order statistic is the max of the partition left of the selected mid.
            let local_bg = if annulus_vals.is_empty() {
                0.0_f64
            } else {
                let m = annulus_vals.len();
                let mid = m / 2;
                let (lower, nth, _) =
                    annulus_vals.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
                if m.is_multiple_of(2) {
                    let prev = lower.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    (prev + *nth) as f64 / 2.0
                } else {
                    *nth as f64
                }
            };

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

            // --- Quadratic peak refinement ---
            let mut final_x = xbar;
            let mut final_y = ybar;

            let pc = peak_col;
            let pr = peak_row;
            if pixel_count >= 5 && pc >= 1 && pr >= 1 && pc + 1 < w && pr + 1 < h {
                // Build 3x3 grid of background-subtracted values around peak
                let effective_bg = local_bg;
                let v = |dy: isize, dx: isize| -> f64 {
                    let r = (pr as isize + dy) as usize;
                    let c = (pc as isize + dx) as usize;
                    gray[r * w + c] as f64 - effective_bg
                };

                if let Some((x_off, y_off)) = quadratic_peak_offset(v) {
                    let qx = pc as f64 + x_off;
                    let qy = pr as f64 + y_off;
                    // Only use quadratic when it agrees with CoM (within 0.5 px).
                    // For asymmetric or blended blobs, CoM is more reliable.
                    let dist_sq = (qx - xbar) * (qx - xbar) + (qy - ybar) * (qy - ybar);
                    if dist_sq < 0.25 {
                        final_x = qx;
                        final_y = qy;
                    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ccl_rejects_degenerate_geometry() {
        let cfg = CentroidExtractionConfig::default();
        // Zero-size and 1-wide images used to panic (chunk size 0 / width-1
        // underflow) rather than return an error.
        assert!(extract_centroids_from_raw(&[], 0, 0, &cfg).is_err());
        assert!(extract_centroids_from_raw(&[1.0], 1, 1, &cfg).is_err());
    }

    #[test]
    fn test_ccl_rejects_bad_config() {
        let pixels = vec![0.0_f32; 16 * 16];
        let zero_block = CentroidExtractionConfig {
            local_bg_block_size: Some(0),
            ..Default::default()
        };
        assert!(extract_centroids_from_raw(&pixels, 16, 16, &zero_block).is_err());
        let nan_thresh = CentroidExtractionConfig {
            sigma_threshold: f32::NAN,
            ..Default::default()
        };
        assert!(extract_centroids_from_raw(&pixels, 16, 16, &nan_thresh).is_err());
    }

    #[test]
    fn test_background_estimation() {
        // Uniform image: background should be ~100, sigma ~0
        let pixels = vec![100.0_f32; 100 * 100];
        let config = CentroidExtractionConfig::default();
        let (mean, sigma) = estimate_background(&pixels, 100, 100, &config);
        assert!((mean - 100.0).abs() < 1.0);
        assert!(sigma < 1.0);
    }

    #[test]
    fn test_extract_from_raw_single_star() {
        let width = 64u32;
        let height = 64u32;
        let mut pixels = vec![10.0_f32; (width * height) as usize];

        // Place a bright Gaussian-ish star near center
        let star_x = 32.0_f32;
        let star_y = 32.0_f32;
        let sigma_px = 2.0_f32;
        for row in 0..height {
            for col in 0..width {
                let dx = col as f32 - star_x;
                let dy = row as f32 - star_y;
                let r2 = dx * dx + dy * dy;
                pixels[(row * width + col) as usize] +=
                    1000.0 * (-r2 / (2.0 * sigma_px * sigma_px)).exp();
            }
        }

        let config = CentroidExtractionConfig {
            sigma_threshold: 3.0,
            min_pixels: 2,
            ..Default::default()
        };

        let result = extract_centroids_from_raw(&pixels, width, height, &config).unwrap();
        assert_eq!(result.centroids.len(), 1);

        // The centroid should be near the center of the image (0, 0 in pixel coords)
        let c = &result.centroids[0];
        assert!(c.x.abs() < 1.0, "Expected x near 0, got {}", c.x);
        assert!(c.y.abs() < 1.0, "Expected y near 0, got {}", c.y);
        assert!(c.mass.unwrap() > 0.0);
    }

    /// Helper: render Gaussian stars on a background with an optional gradient
    /// and deterministic (seedless) per-pixel noise of amplitude `noise`.
    fn render_stars(
        width: u32,
        height: u32,
        bg: f32,
        gradient: f32,
        noise: f32,
        sigma_px: f32,
        stars: &[(f32, f32, f32)],
    ) -> Vec<f32> {
        let (w, h) = (width as usize, height as usize);
        let mut pixels = vec![0.0_f32; w * h];
        for row in 0..h {
            for col in 0..w {
                // Large-scale gradient the coarse-grid background must reject,
                // plus a cheap deterministic dither so the noise σ is nonzero.
                let dither = (((row * w + col) as u32).wrapping_mul(2_654_435_761) >> 8) as f32
                    / 16_777_216.0
                    - 0.5;
                pixels[row * w + col] = bg + gradient * (col as f32 / w as f32) + noise * dither;
            }
        }
        for &(sx, sy, brightness) in stars {
            for row in 0..h {
                for col in 0..w {
                    let dx = col as f32 - sx;
                    let dy = row as f32 - sy;
                    let r2 = dx * dx + dy * dy;
                    pixels[row * w + col] += brightness * (-r2 / (2.0 * sigma_px * sigma_px)).exp();
                }
            }
        }
        pixels
    }

    #[test]
    fn test_fast_extract_recovers_stars_over_gradient() {
        let (width, height) = (128u32, 128u32);
        let sigma_px = 1.6_f32;
        // Sub-pixel true positions; a strong left-to-right gradient the
        // coarse-grid background must track, plus realistic noise.
        let stars = [
            (30.3, 30.0, 900.0),
            (90.0, 50.7, 1300.0),
            (60.5, 100.2, 600.0),
        ];
        let pixels = render_stars(width, height, 50.0, 400.0, 8.0, sigma_px, &stars);

        let config = FastCentroidConfig {
            sigma_threshold: 5.0,
            bg_grid: 32,
            ..Default::default()
        };
        let result = extract_centroids_fast(&pixels, width, height, &config).unwrap();
        assert_eq!(
            result.centroids.len(),
            3,
            "expected 3 stars, got {}",
            result.centroids.len()
        );
        // Brightest-first ordering.
        assert!(result.centroids[0].mass.unwrap() >= result.centroids[1].mass.unwrap());

        // Each true star must have a detection within ~0.6 px. The single-pass
        // path is a ~0.5-px-class centroider by design (threshold-clipped CoM +
        // parabola refine) — plenty for solving, not for tight astrometry.
        let cx = (width - 1) as f32 / 2.0;
        let cy = (height - 1) as f32 / 2.0;
        for &(sx, sy, _) in &stars {
            let (tx, ty) = (sx - cx, sy - cy);
            let best = result
                .centroids
                .iter()
                .map(|c| ((c.x - tx).powi(2) + (c.y - ty).powi(2)).sqrt())
                .fold(f32::INFINITY, f32::min);
            assert!(
                best < 0.6,
                "star ({sx}, {sy}) nearest detection {best:.3} px away"
            );
        }
    }

    #[test]
    fn test_fast_extract_merges_touching_pixels_and_caps() {
        let (width, height) = (128u32, 128u32);
        // Two stars 1 px apart form one connected region (correct for a blended
        // pair); a far star is its own region → 2 total.
        let stars = [
            (64.0, 64.0, 1000.0),
            (65.0, 64.0, 950.0),
            (20.0, 20.0, 800.0),
        ];
        let pixels = render_stars(width, height, 30.0, 0.0, 6.0, 1.5, &stars);

        let config = FastCentroidConfig {
            sigma_threshold: 5.0,
            max_centroids: Some(5),
            ..Default::default()
        };
        let result = extract_centroids_fast(&pixels, width, height, &config).unwrap();
        assert_eq!(
            result.centroids.len(),
            2,
            "blended pair should merge to 1 + 1 separate = 2, got {}",
            result.centroids.len()
        );
    }

    #[test]
    fn test_fast_extract_rejects_bad_params() {
        let pixels = vec![0.0_f32; 64 * 64];
        let bad_sigma = FastCentroidConfig {
            sigma_threshold: 0.0,
            ..Default::default()
        };
        assert!(extract_centroids_fast(&pixels, 64, 64, &bad_sigma).is_err());
        let bad_grid = FastCentroidConfig {
            bg_grid: 0,
            ..Default::default()
        };
        assert!(extract_centroids_fast(&pixels, 64, 64, &bad_grid).is_err());
        // Length mismatch.
        assert!(extract_centroids_fast(&pixels, 64, 63, &FastCentroidConfig::default()).is_err());
    }

    #[test]
    fn test_extract_from_raw_multiple_stars() {
        let width = 128u32;
        let height = 128u32;
        let mut pixels = vec![10.0_f32; (width * height) as usize];

        // Place 3 stars at different positions
        let stars = [
            (30.0, 30.0, 800.0),
            (90.0, 50.0, 1200.0),
            (60.0, 100.0, 500.0),
        ];
        let sigma_px = 2.0_f32;

        for &(sx, sy, brightness) in &stars {
            for row in 0..height {
                for col in 0..width {
                    let dx = col as f32 - sx;
                    let dy = row as f32 - sy;
                    let r2 = dx * dx + dy * dy;
                    pixels[(row * width + col) as usize] +=
                        brightness * (-r2 / (2.0 * sigma_px * sigma_px)).exp();
                }
            }
        }

        let config = CentroidExtractionConfig {
            sigma_threshold: 3.0,
            min_pixels: 2,
            ..Default::default()
        };

        let result = extract_centroids_from_raw(&pixels, width, height, &config).unwrap();
        assert_eq!(
            result.centroids.len(),
            3,
            "Expected 3 stars, got {}",
            result.centroids.len()
        );

        // Centroids should be sorted by brightness (descending)
        assert!(result.centroids[0].mass.unwrap() >= result.centroids[1].mass.unwrap());
        assert!(result.centroids[1].mass.unwrap() >= result.centroids[2].mass.unwrap());
    }

    #[test]
    fn test_max_centroids_limit() {
        let width = 128u32;
        let height = 128u32;
        let mut pixels = vec![10.0_f32; (width * height) as usize];

        let stars = [
            (30.0, 30.0, 800.0),
            (90.0, 50.0, 1200.0),
            (60.0, 100.0, 500.0),
        ];
        let sigma_px = 2.0_f32;

        for &(sx, sy, brightness) in &stars {
            for row in 0..height {
                for col in 0..width {
                    let dx = col as f32 - sx;
                    let dy = row as f32 - sy;
                    let r2 = dx * dx + dy * dy;
                    pixels[(row * width + col) as usize] +=
                        brightness * (-r2 / (2.0 * sigma_px * sigma_px)).exp();
                }
            }
        }

        let config = CentroidExtractionConfig {
            sigma_threshold: 3.0,
            min_pixels: 2,
            max_centroids: Some(2),
            ..Default::default()
        };

        let result = extract_centroids_from_raw(&pixels, width, height, &config).unwrap();
        assert_eq!(result.centroids.len(), 2);
    }

    #[test]
    fn test_quadratic_refinement() {
        // Place a Gaussian star at a known sub-pixel offset on uniform background
        let width = 64u32;
        let height = 64u32;
        let bg = 100.0_f32;
        let true_x = 32.3_f32;
        let true_y = 32.7_f32;
        let sigma_px = 2.0_f32;
        let peak_brightness = 2000.0_f32;

        let mut pixels = vec![bg; (width * height) as usize];
        for row in 0..height {
            for col in 0..width {
                let dx = col as f32 - true_x;
                let dy = row as f32 - true_y;
                let r2 = dx * dx + dy * dy;
                pixels[(row * width + col) as usize] +=
                    peak_brightness * (-r2 / (2.0 * sigma_px * sigma_px)).exp();
            }
        }

        let config = CentroidExtractionConfig {
            sigma_threshold: 3.0,
            min_pixels: 3,
            ..Default::default()
        };

        let result = extract_centroids_from_raw(&pixels, width, height, &config).unwrap();
        assert_eq!(
            result.centroids.len(),
            1,
            "Expected 1 star, got {}",
            result.centroids.len()
        );

        // Centroid is in centered coords (origin at image center)
        let c = &result.centroids[0];
        let cx = (width - 1) as f32 / 2.0;
        let cy = (height - 1) as f32 / 2.0;
        let abs_x = c.x + cx;
        let abs_y = c.y + cy;

        let err_x = (abs_x - true_x).abs();
        let err_y = (abs_y - true_y).abs();
        assert!(
            err_x < 0.15,
            "X error too large: centroid={abs_x:.4}, true={true_x}, err={err_x:.4}"
        );
        assert!(
            err_y < 0.15,
            "Y error too large: centroid={abs_y:.4}, true={true_y}, err={err_y:.4}"
        );
    }

    #[test]
    fn test_quadratic_refinement_with_gradient_background() {
        // Place a star on a gradient background to test local background correction
        let width = 128u32;
        let height = 128u32;
        let true_x = 64.4_f32;
        let true_y = 64.6_f32;
        let sigma_px = 2.0_f32;
        let peak_brightness = 2000.0_f32;

        let mut pixels = vec![0.0_f32; (width * height) as usize];
        // Add a gradient background: increases from left to right (50 to 150)
        for row in 0..height {
            for col in 0..width {
                let bg = 50.0 + 100.0 * (col as f32 / width as f32);
                pixels[(row * width + col) as usize] = bg;
            }
        }
        // Add Gaussian star
        for row in 0..height {
            for col in 0..width {
                let dx = col as f32 - true_x;
                let dy = row as f32 - true_y;
                let r2 = dx * dx + dy * dy;
                pixels[(row * width + col) as usize] +=
                    peak_brightness * (-r2 / (2.0 * sigma_px * sigma_px)).exp();
            }
        }

        let config = CentroidExtractionConfig {
            sigma_threshold: 5.0,
            min_pixels: 3,
            ..Default::default()
        };

        let result = extract_centroids_from_raw(&pixels, width, height, &config).unwrap();
        assert!(
            !result.centroids.is_empty(),
            "Should detect at least one star on gradient background"
        );

        // Find the centroid closest to our true position
        let cx = (width - 1) as f32 / 2.0;
        let cy = (height - 1) as f32 / 2.0;
        let best = result
            .centroids
            .iter()
            .min_by(|a, b| {
                let da = (a.x + cx - true_x).powi(2) + (a.y + cy - true_y).powi(2);
                let db = (b.x + cx - true_x).powi(2) + (b.y + cy - true_y).powi(2);
                da.partial_cmp(&db).unwrap()
            })
            .unwrap();

        let abs_x = best.x + cx;
        let abs_y = best.y + cy;
        let err_x = (abs_x - true_x).abs();
        let err_y = (abs_y - true_y).abs();
        assert!(
            err_x < 0.3,
            "X error too large on gradient bg: centroid={abs_x:.4}, true={true_x}, err={err_x:.4}"
        );
        assert!(
            err_y < 0.3,
            "Y error too large on gradient bg: centroid={abs_y:.4}, true={true_y}, err={err_y:.4}"
        );
    }
}
