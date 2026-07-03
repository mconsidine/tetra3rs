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

mod ccl;
mod fast;

pub use fast::{extract_centroids_fast, FastCentroidConfig};

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
    /// A matched filter boosts point-source SNR before thresholding —
    /// ~2× peak SNR (≈0.75 mag more depth at the same false-positive rate)
    /// for a σ≈1.5 px PSF. The gain is largest for faint stars in noisy or
    /// dense images, and the optimum is broad: σ within a factor of ~2 of
    /// the true PSF width recovers nearly all of it.
    ///
    /// The detection threshold is automatically scaled by the kernel's
    /// noise-suppression factor, so `sigma_threshold` means "sigmas of the
    /// noise actually present in the thresholded image" whether the filter
    /// is on or off — no retuning needed when toggling it.
    ///
    /// Default: Some(1.5). Set `None` to threshold the unfiltered image
    /// (marginally faster; appropriate when downstream limits like
    /// `max_centroids` make faint-star depth irrelevant).
    pub matched_filter_sigma: Option<f32>,

    /// Maximum DAOFIND-style sharpness: `(peak − mean(8 neighbors)) / peak`,
    /// measured on the background-subtracted image at the blob's peak. Values
    /// near 1 mean the flux is concentrated in a single pixel — a hot pixel
    /// or cosmic-ray hit rather than a star. A critically sampled PSF scores
    /// ~0.5; a strongly undersampled one can reach ~0.85. The default 0.9
    /// passes any system whose PSF spans multiple pixels (the design norm —
    /// star trackers defocus deliberately, because a sub-pixel PSF forfeits
    /// sub-pixel centroiding). Set `None` for severely undersampled data
    /// (PSF FWHM below ~1.5 px), where real stars are geometrically
    /// indistinguishable from hot pixels.
    ///
    /// Default: Some(0.9)
    pub max_sharpness: Option<f32>,

    /// Pixel value at or above which the sensor is considered saturated.
    /// A blob whose peak reaches this level skips quadratic peak refinement
    /// (a flat-topped or bloomed profile has no meaningful sub-pixel
    /// maximum), keeping the center-of-mass position instead.
    ///
    /// Default: None (disabled)
    pub saturation_level: Option<f32>,
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
            matched_filter_sigma: Some(1.5),
            max_sharpness: Some(0.9),
            saturation_level: None,
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

    /// Number of blobs found before the size/elongation filters are applied
    /// (connected components on the CCL path; detected regions before the
    /// `min_pixels` filter on the fast path).
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
    ccl::extract_from_gray(&gray, width, height, config)
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
    check_pixel_len(pixels.len(), width, height)?;
    ccl::extract_from_gray(pixels, width, height, config)
}

// ─── Internal helpers ──────────────────────────────────────────────────────

/// Upper-midpoint order statistic `v[len/2]` — the extraction pipeline's cheap
/// "median" convention for background grids. O(n) selection (partitions the
/// slice in place); `0.0` for an empty slice.
fn midpoint_f32(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let m = values.len() / 2;
    let (_, nth, _) = values.select_nth_unstable_by(m, |a, b| a.total_cmp(b));
    *nth
}

/// Median of the values (partitioned in place, O(n) selection): even lengths
/// average the two central order statistics — `values[n/2]` (the selected
/// element) and `values[n/2 − 1]` (the max of the lower partition that
/// `select_nth` leaves to its left). `0.0` for an empty slice.
fn median_f32(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let n = values.len();
    let (lower, nth, _) = values.select_nth_unstable_by(n / 2, |a, b| a.total_cmp(b));
    if n.is_multiple_of(2) {
        let prev = lower.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        (prev + *nth) / 2.0
    } else {
        *nth
    }
}

/// Sort centroids brightest-first (descending mass; missing mass sorts as 0)
/// and truncate to the configured maximum. Shared tail of both extraction
/// paths.
fn sort_and_truncate_by_mass(centroids: &mut Vec<Centroid>, max_centroids: Option<usize>) {
    centroids.sort_by(|a, b| {
        b.mass
            .unwrap_or(0.0)
            .partial_cmp(&a.mass.unwrap_or(0.0))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if let Some(max) = max_centroids {
        centroids.truncate(max);
    }
}

/// Validate that a raw pixel buffer matches the claimed dimensions.
fn check_pixel_len(len: usize, width: u32, height: u32) -> Result<()> {
    let expected = (width as usize) * (height as usize);
    if len != expected {
        return Err(Error::InvalidInput(format!(
            "Pixel data length ({len}) does not match width*height ({width}x{height}={expected})",
        )));
    }
    Ok(())
}

/// 3×3 parabola sub-pixel refinement at the integer peak `(pc, pr)`, gated the
/// same way in both extraction paths: the blob must have ≥ 5 pixels, the peak
/// must not touch the border of the `(w, h)` image, and the fitted position
/// must agree with the center-of-mass estimate `(com_x, com_y)` within 0.5 px
/// (for asymmetric or blended blobs the CoM is more reliable). Returns the
/// refined position, or `None` to keep the CoM.
fn accepted_peak_refine(
    npix: usize,
    (pc, pr): (usize, usize),
    (w, h): (usize, usize),
    (com_x, com_y): (f64, f64),
    v: impl Fn(isize, isize) -> f64,
) -> Option<(f64, f64)> {
    if npix < 5 || pc < 1 || pr < 1 || pc + 1 >= w || pr + 1 >= h {
        return None;
    }
    let (x_off, y_off) = quadratic_peak_offset(v)?;
    let qx = pc as f64 + x_off;
    let qy = pr as f64 + y_off;
    let dist_sq = (qx - com_x) * (qx - com_x) + (qy - com_y) * (qy - com_y);
    if dist_sq < 0.25 {
        Some((qx, qy))
    } else {
        None
    }
}

/// DAOFIND-style sharpness of a blob peak: `(peak − mean(8 neighbors)) / peak`
/// on background-subtracted values (`v(dy, dx)` samples relative to the peak,
/// the same accessor convention as [`accepted_peak_refine`]). Out-of-bounds
/// neighbors are skipped. Values near 1 mean the flux is concentrated in a
/// single pixel — a hot pixel or cosmic-ray hit; a real PSF puts substantial
/// flux into the neighbors (critically sampled ~0.5, strongly undersampled up
/// to ~0.85). Returns `None` when the peak is non-positive or has no
/// in-bounds neighbors (sharpness undefined — callers should not reject).
fn peak_sharpness(
    (pc, pr): (usize, usize),
    (w, h): (usize, usize),
    v: impl Fn(isize, isize) -> f64,
) -> Option<f64> {
    let peak = v(0, 0);
    if peak <= 0.0 {
        return None;
    }
    let mut sum = 0.0_f64;
    let mut n = 0u32;
    for dy in -1..=1_isize {
        for dx in -1..=1_isize {
            if dy == 0 && dx == 0 {
                continue;
            }
            let rr = pr as isize + dy;
            let cc = pc as isize + dx;
            if rr < 0 || cc < 0 || rr >= h as isize || cc >= w as isize {
                continue;
            }
            sum += v(dy, dx);
            n += 1;
        }
    }
    if n == 0 {
        return None;
    }
    Some((peak - sum / n as f64) / peak)
}

/// Convert a DynamicImage to a Vec<f32> of grayscale values.
fn to_grayscale_f32(img: &image::DynamicImage) -> Vec<f32> {
    use image::DynamicImage;
    match img {
        // 16-bit images: cast to f32 (values keep their native [0, 65535] range)
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

#[cfg(test)]
mod tests {
    use super::ccl::estimate_background;
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

    #[test]
    fn test_matched_filter_depth_gain() {
        // A star too faint for the unfiltered 5σ cut is recovered when the
        // matched filter is on — at the SAME sigma_threshold, because the
        // detection threshold is scaled by the kernel's noise-suppression
        // factor automatically.
        let (width, height) = (64u32, 64u32);
        let pixels = render_stars(width, height, 100.0, 0.0, 20.0, 1.5, &[(30.0, 30.0, 20.0)]);
        let base = CentroidExtractionConfig {
            sigma_threshold: 5.0,
            local_bg_block_size: None,
            matched_filter_sigma: None,
            ..Default::default()
        };
        let unfiltered = extract_centroids_from_raw(&pixels, width, height, &base).unwrap();
        assert_eq!(
            unfiltered.centroids.len(),
            0,
            "star should sit below the unfiltered cut"
        );

        let filtered_cfg = CentroidExtractionConfig {
            matched_filter_sigma: Some(1.5),
            ..base
        };
        let filtered = extract_centroids_from_raw(&pixels, width, height, &filtered_cfg).unwrap();
        assert_eq!(
            filtered.centroids.len(),
            1,
            "matched filter should recover the faint star"
        );
        assert!((filtered.centroids[0].x - (30.0 - 31.5)).abs() < 1.0);
        assert!((filtered.centroids[0].y - (30.0 - 31.5)).abs() < 1.0);
    }

    #[test]
    fn test_matched_filter_no_noise_false_positives() {
        // Pure noise + gradient with the (default-on) filter and local
        // background: the compensated threshold must keep false positives at
        // zero. Regression guard: convolving the *clamped* residual rectified
        // negative noise into a positive DC offset comparable to the
        // compensated threshold, which would light up the whole frame.
        let (width, height) = (128u32, 128u32);
        let pixels = render_stars(width, height, 100.0, 30.0, 20.0, 1.5, &[]);
        let cfg = CentroidExtractionConfig {
            sigma_threshold: 5.0,
            ..Default::default()
        };
        let res = extract_centroids_from_raw(&pixels, width, height, &cfg).unwrap();
        assert_eq!(
            res.centroids.len(),
            0,
            "noise-only frame produced detections"
        );
    }

    #[test]
    fn test_peak_sharpness_values() {
        // Lone hot pixel: all 8 neighbors zero → sharpness exactly 1.
        let hot = |dy: isize, dx: isize| if dy == 0 && dx == 0 { 100.0 } else { 0.0 };
        assert_eq!(peak_sharpness((1, 1), (3, 3), hot), Some(1.0));
        // Flat plateau: neighbors equal the peak → sharpness 0.
        let flat = |_: isize, _: isize| 50.0;
        assert_eq!(peak_sharpness((1, 1), (3, 3), flat), Some(0.0));
        // Corner peak: only the 3 in-bounds neighbors are averaged.
        let corner = |dy: isize, dx: isize| if dy == 0 && dx == 0 { 90.0 } else { 30.0 };
        assert_eq!(
            peak_sharpness((0, 0), (3, 3), corner),
            Some((90.0 - 30.0) / 90.0)
        );
        // Non-positive peak: undefined.
        assert_eq!(peak_sharpness((1, 1), (3, 3), |_, _| -1.0), None);
    }

    #[test]
    fn test_sharpness_gate_rejects_hot_pixel() {
        // A real star plus a single hot pixel. The matched filter smears the
        // hot pixel into a blob that passes `min_pixels`, but its sharpness
        // on the *unfiltered* image (~1.0) trips the gate; the star (~0.5)
        // survives. With the gate disabled, both are detected.
        let (width, height) = (64u32, 64u32);
        let mut pixels = render_stars(width, height, 10.0, 0.0, 2.0, 1.5, &[(20.0, 20.0, 800.0)]);
        pixels[44 * 64 + 44] += 1200.0;

        let base = CentroidExtractionConfig {
            sigma_threshold: 4.0,
            min_pixels: 3,
            matched_filter_sigma: Some(1.5),
            local_bg_block_size: None,
            max_sharpness: Some(0.9),
            ..Default::default()
        };
        let gated = extract_centroids_from_raw(&pixels, width, height, &base).unwrap();
        assert_eq!(
            gated.centroids.len(),
            1,
            "hot pixel should be rejected by the sharpness gate"
        );
        assert!((gated.centroids[0].x - (20.0 - 31.5)).abs() < 1.0);

        let ungated = CentroidExtractionConfig {
            max_sharpness: None,
            ..base
        };
        let all = extract_centroids_from_raw(&pixels, width, height, &ungated).unwrap();
        assert_eq!(
            all.centroids.len(),
            2,
            "gate disabled: hot pixel should be detected"
        );
    }

    #[test]
    fn test_fast_path_sharpness_gate() {
        // Single hot pixel with min_pixels = 1: only the sharpness gate can
        // reject it on the fast path.
        let (width, height) = (64u32, 64u32);
        let mut pixels = render_stars(width, height, 10.0, 0.0, 2.0, 1.5, &[(20.0, 20.0, 800.0)]);
        pixels[44 * 64 + 44] += 1200.0;

        let base = FastCentroidConfig {
            sigma_threshold: 4.0,
            min_pixels: 1,
            max_sharpness: Some(0.9),
            ..Default::default()
        };
        let gated = extract_centroids_fast(&pixels, width, height, &base).unwrap();
        assert_eq!(gated.centroids.len(), 1, "hot pixel should be rejected");

        let ungated = FastCentroidConfig {
            max_sharpness: None,
            ..base
        };
        let all = extract_centroids_fast(&pixels, width, height, &ungated).unwrap();
        assert_eq!(all.centroids.len(), 2, "gate disabled: hot pixel detected");
    }

    #[test]
    fn test_saturation_guard_keeps_com() {
        // A clipped (flat-top) star: with `saturation_level` set the parabola
        // refinement is skipped and the CoM position is kept. The symmetric
        // clipped PSF still centroids onto the true position.
        let (width, height) = (64u32, 64u32);
        let raw = render_stars(width, height, 10.0, 0.0, 1.0, 2.0, &[(30.0, 33.0, 20000.0)]);
        let clipped: Vec<f32> = raw.iter().map(|&v| v.min(1000.0)).collect();

        let config = CentroidExtractionConfig {
            sigma_threshold: 4.0,
            saturation_level: Some(1000.0),
            local_bg_block_size: None,
            ..Default::default()
        };
        let res = extract_centroids_from_raw(&clipped, width, height, &config).unwrap();
        assert_eq!(res.centroids.len(), 1);
        let c = &res.centroids[0];
        assert!(
            (c.x - (30.0 - 31.5)).abs() < 0.3 && (c.y - (33.0 - 31.5)).abs() < 0.3,
            "saturated star CoM off: ({}, {})",
            c.x,
            c.y
        );
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
