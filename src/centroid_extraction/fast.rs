//! Fast single-pass star-tracker extraction path: coarse subsampled
//! background grid + one raster sweep with run-length connected regions and
//! inline moment accumulation. Split out of the crate-facing module; entry via
//! [`extract_centroids_fast`].

use super::{
    accepted_peak_refine, check_pixel_len, midpoint_f32, peak_sharpness, sort_and_truncate_by_mass,
    CentroidExtractionResult,
};
use crate::centroid::Centroid;
use crate::error::{Error, Result};

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

    /// Maximum DAOFIND-style sharpness: `(peak − mean(8 neighbors)) / peak`,
    /// measured on the background-subtracted image at the region's peak.
    /// Values near 1 mean single-pixel flux — a hot pixel or cosmic-ray hit.
    /// A critically sampled PSF scores ~0.5; a strongly undersampled one up
    /// to ~0.85. The default 0.9 passes any system whose PSF spans multiple
    /// pixels; set `None` for severely undersampled data (PSF FWHM below
    /// ~1.5 px), where real stars are indistinguishable from hot pixels.
    /// Default: Some(0.9)
    pub max_sharpness: Option<f32>,

    /// Pixel value at or above which the sensor is considered saturated.
    /// A region whose peak reaches this level skips the 3×3 parabola
    /// refinement (a flat-topped profile has no meaningful sub-pixel
    /// maximum), keeping the center-of-mass position.
    /// Default: None (disabled)
    pub saturation_level: Option<f32>,
}

impl Default for FastCentroidConfig {
    fn default() -> Self {
        Self {
            sigma_threshold: 5.0,
            bg_grid: 64,
            min_pixels: 2,
            max_centroids: None,
            max_sharpness: Some(0.9),
            saturation_level: None,
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
    check_pixel_len(pixels.len(), width, height)?;
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

        let (pc, pr) = (reg.peak_x as usize, reg.peak_y as usize);
        let bg = bilinear_grid(&bg_grid, nx, ny, block, pc, pr) as f64;
        let v = |dy: isize, dx: isize| -> f64 {
            let rr = (pr as isize + dy) as usize;
            let cc = (pc as isize + dx) as usize;
            pixels[rr * w + cc] as f64 - bg
        };

        // Hot-pixel / cosmic-ray sharpness gate (shared with the CCL path).
        if let Some(max_sharp) = config.max_sharpness {
            if let Some(s) = peak_sharpness((pc, pr), (w, h), v) {
                if s > max_sharp as f64 {
                    continue;
                }
            }
        }

        // Optional 3×3 parabola refine on the raw image at the peak (shared
        // gate with the CCL path — see `accepted_peak_refine`). Skipped for
        // saturated peaks (no meaningful sub-pixel maximum on a flat top).
        let saturated = config.saturation_level.is_some_and(|s| reg.peak_val >= s);
        if !saturated {
            if let Some((qx, qy)) =
                accepted_peak_refine(reg.npix as usize, (pc, pr), (w, h), (fx, fy), v)
            {
                fx = qx;
                fy = qy;
            }
        }

        centroids.push(Centroid {
            x: fx as f32 - cx,
            y: fy as f32 - cy,
            mass: Some(reg.sum_w as f32),
            cov: None,
        });
    }

    sort_and_truncate_by_mass(&mut centroids, config.max_centroids);

    let bg_mean = midpoint_f32(&mut bg_grid.clone());

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
            let median = midpoint_f32(&mut samples);
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
