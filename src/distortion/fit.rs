//! Distortion model fitting from solve results.
//!
//! Given one or more plate-solve results (each with matched catalog star IDs and
//! centroid indices), this module fits distortion models by comparing observed
//! centroid positions to their ideal (pinhole-projected) positions computed from
//! the catalog.
//!
//! The fitting is iterative with sigma-clipping to reject mismatched stars.

use std::collections::HashMap;

use numeris::optim::{least_squares_lm_dyn, LmSettings};
use numeris::{DynMatrix, DynVector, Matrix3};
use tracing::debug;

use crate::centroid::Centroid;
use crate::solver::{SolveResult, SolverDatabase};

use super::polynomial::{num_coeffs, term_pairs, PolynomialDistortion};
use super::radial::RadialDistortion;
use super::Distortion;

/// Minimum matched points for a Brown-Conrady radial fit. The LM solves for 8
/// parameters (cx, cy, γ, k1–k3, p1, p2), so fewer points is under-determined —
/// [`run_intrinsics_lm`] bails below this and callers must not treat the
/// resulting warm-start as a real fit.
pub(super) const MIN_RADIAL_POINTS: usize = 8;

/// Configuration for distortion fitting.
#[derive(Debug, Clone)]
pub struct DistortionFitConfig {
    /// Sigma threshold for iterative outlier rejection. Default 3.0.
    pub sigma_clip: f64,
    /// Maximum iterations for iterative fitting. Default 20.
    pub max_iterations: u32,
    /// If provided, a second stage re-applies the model with this loose pixel
    /// threshold to recover stars rejected in the tight sigma-clip stage.
    /// Stars with residuals below this threshold are kept. Default 5.0 px.
    pub stage2_threshold_px: Option<f64>,
}

impl Default for DistortionFitConfig {
    fn default() -> Self {
        Self {
            sigma_clip: 3.0,
            max_iterations: 20,
            stage2_threshold_px: Some(5.0),
        }
    }
}

/// Result of a distortion fitting procedure.
#[derive(Debug, Clone)]
pub struct DistortionFitResult {
    /// The fitted distortion model.
    pub model: Distortion,
    /// Optional optical center offset (CRPIX) fit jointly with the
    /// distortion model.
    ///
    /// - `None` for [`Distortion::Polynomial`] fits — the polynomial
    ///   absorbs the center offset into its order-0 (constant) terms,
    ///   which `extract_crpix` later moves into [`CameraModel::crpix`].
    /// - `None` for [`Distortion::Radial`] fits too — the jointly-fit
    ///   optical-axis position is carried inside the model itself
    ///   ([`RadialDistortion::center`]), NOT in `crpix`: on mosaic cameras
    ///   the optical axis can sit far off the detector (a CCD corner on
    ///   TESS), while `crpix` is the projection origin and must stay near
    ///   the image center for the solver's geometry.
    pub crpix: Option<[f64; 2]>,
    /// Correction factor for the anchor focal length (the focal length
    /// implied by the solve FOV used to project the ideal points):
    /// `f_true = focal_scale · f_anchor`. Always `1.0` for polynomial fits;
    /// for radial fits this carries the jointly-fit linear scale term (the
    /// Brown-Conrady model itself has no linear degree of freedom).
    pub focal_scale: f64,
    /// RMS residual in pixels BEFORE distortion correction.
    pub rmse_before_px: f64,
    /// RMS residual in pixels AFTER distortion correction.
    pub rmse_after_px: f64,
    /// Number of inlier matches used in the final fit.
    pub n_inliers: usize,
    /// Number of outlier matches rejected.
    pub n_outliers: usize,
    /// Number of sigma-clip iterations performed.
    pub iterations: u32,
}

// ── Data structures for internal use ────────────────────────────────────────

/// A single matched observation: observed centroid pixel position + ideal (projected) position.
pub(super) struct MatchedPoint {
    /// Observed centroid x (distorted), pixels from image center.
    pub x_obs: f64,
    /// Observed centroid y (distorted), pixels from image center.
    pub y_obs: f64,
    /// Ideal projected x (undistorted pinhole model), pixels from image center.
    pub x_ideal: f64,
    /// Ideal projected y (undistorted pinhole model), pixels from image center.
    pub y_ideal: f64,
}

// ── Radial distortion fitting ───────────────────────────────────────────────

/// Fit a radial Brown-Conrady distortion model `(k1, k2, k3)` plus optical
/// center `(cx, cy)` from plate-solve results.
///
/// Joint nonlinear least squares (Gauss-Newton, MAD sigma-clipped). The
/// model is OpenCV-style: the radial term is centered on the fitted optical
/// axis, not on the geometric image center —
///
/// ```text
///     x_n = x_ideal − cx,   y_n = y_ideal − cy
///     r²  = x_n² + y_n²
///     x_obs = cx + x_n · (1 + k1·r² + k2·r⁴ + k3·r⁶)
///     y_obs = cy + y_n · (1 + k1·r² + k2·r⁴ + k3·r⁶)
/// ```
///
/// Returns the radial coefficients in [`DistortionFitResult::model`] and
/// the fitted optical-center offset in [`DistortionFitResult::crpix`].
/// Same interface as [`fit_polynomial_distortion`] otherwise. Suitable for
/// most photographic / computer-vision lens calibrations; for cameras with
/// significant tangential / decentering distortion (e.g. TESS), prefer
/// [`fit_polynomial_distortion`] which has more parameters to absorb it.
pub fn fit_radial_distortion(
    solve_results: &[&SolveResult],
    centroids: &[&[Centroid]],
    database: &SolverDatabase,
    image_width: u32,
    config: &DistortionFitConfig,
) -> DistortionFitResult {
    assert_eq!(
        solve_results.len(),
        centroids.len(),
        "solve_results and centroids must have the same length"
    );

    let id_to_idx = build_id_lookup(database);
    let points = gather_matched_points(solve_results, centroids, database, &id_to_idx, image_width);

    // Below MIN_RADIAL_POINTS the 8-parameter fit is under-determined and the LM
    // returns its (identity) warm start, which would masquerade as a real fit.
    // Report a clean no-op result instead.
    if points.len() < MIN_RADIAL_POINTS {
        return DistortionFitResult {
            model: Distortion::None,
            crpix: None,
            focal_scale: 1.0,
            rmse_before_px: 0.0,
            rmse_after_px: 0.0,
            n_inliers: 0,
            n_outliers: 0,
            iterations: 0,
        };
    }

    let n = points.len();
    let fit = fit_radial_centered_sigma_clip(&points, config);
    // Re-express in the scale-corrected frame (γ folded into the focal
    // length via `focal_scale`). Predicted pixels are unchanged, so
    // residuals are evaluated with the raw fitted params including γ.
    let model = fit.rescaled_model();

    // Compute before/after RMSE on the SAME inlier set for a fair comparison.
    let rmse_before =
        compute_corrected_rmse_centered(&points, &fit.mask, &Distortion::None, [0.0, 0.0]);
    let residuals = intrinsics_residuals(
        &points,
        &[
            fit.cx, fit.cy, fit.gamma, fit.k1, fit.k2, fit.k3, fit.p1, fit.p2,
        ],
    );
    let rmse_after = masked_rms(&residuals, &fit.mask);
    let n_inliers = fit.mask.iter().filter(|&&m| m).count();

    debug!(
        "Brown-Conrady fit: cx={:.2}, cy={:.2}, gamma={:.6}, k1={:.3e}, k2={:.3e}, k3={:.3e}, p1={:.3e}, p2={:.3e}, inliers={}/{}, RMSE {:.3} → {:.3} px",
        fit.cx, fit.cy, fit.gamma, fit.k1, fit.k2, fit.k3, fit.p1, fit.p2, n_inliers, n, rmse_before, rmse_after
    );

    DistortionFitResult {
        model: Distortion::Radial(model),
        crpix: None,
        focal_scale: fit.gamma,
        rmse_before_px: rmse_before,
        rmse_after_px: rmse_after,
        n_inliers,
        n_outliers: n - n_inliers,
        iterations: fit.iterations,
    }
}

/// Result of a sigma-clipped camera-intrinsics fit: optical center,
/// focal-scale factor, and Brown-Conrady distortion (radial + tangential).
pub(super) struct CenteredRadialFitResult {
    /// Optical-center offset in pixels, in the geometric (no-crpix) frame.
    pub cx: f64,
    pub cy: f64,
    /// Focal-length correction factor: `f_true = gamma · f_anchor`, where
    /// `f_anchor` is the focal length used to project the ideal points.
    /// `f_anchor` derives from the solve FOV — a whole-field average biased
    /// by the very distortion being fit (≈1% on TESS) — and Brown-Conrady
    /// has no linear term, so without this degree of freedom the anchor
    /// bias becomes a linear residual (tens of px at the field corner) that
    /// the cubic+ terms can only mimic.
    pub gamma: f64,
    pub k1: f64,
    pub k2: f64,
    pub k3: f64,
    /// Tangential / decentering coefficients.
    pub p1: f64,
    pub p2: f64,
    pub mask: Vec<bool>,
    pub iterations: u32,
}

impl CenteredRadialFitResult {
    /// The fitted distortion re-expressed in the corrected ideal frame
    /// (focal length × `gamma`): `kᵢ′ = kᵢ/γ^2i`, `p′ = p/γ`, carrying the
    /// fitted optical-axis position as the model's own `center`. Up to a
    /// small constant offset `(γ−1)·c` (absorbed by attitude at solve
    /// time), predicted observed pixels are unchanged.
    pub(super) fn rescaled_model(&self) -> RadialDistortion {
        let g = self.gamma;
        let g2 = g * g;
        RadialDistortion::with_center(
            self.cx,
            self.cy,
            self.k1 / g2,
            self.k2 / (g2 * g2),
            self.k3 / (g2 * g2 * g2),
            self.p1 / g,
            self.p2 / g,
        )
    }
}

/// Joint nonlinear LS fit of the standard camera-intrinsics model — optical
/// center `(cx, cy)`, focal-scale factor `γ`, and Brown-Conrady distortion
/// `(k1, k2, k3, p1, p2)` — via Levenberg-Marquardt with MAD-based
/// sigma-clipping. This is the parameter set OpenCV's `calibrateCamera`
/// fits (free principal point, free focal length, 5-coefficient
/// Brown-Conrady), restricted to square pixels and no skew. Reusable across
/// single-image and multi-image calibration paths.
///
/// The forward model, centered on the fitted optical axis:
///
/// ```text
///     x_n = x_ideal − cx,   y_n = y_ideal − cy
///     r²  = x_n² + y_n²
///     rad = 1 + k1·r² + k2·r⁴ + k3·r⁶
///     x_obs = cx + γ·(x_n·rad + 2·p1·x_n·y_n + p2·(r² + 2·x_n²))
///     y_obs = cy + γ·(y_n·rad + p1·(r² + 2·y_n²) + 2·p2·x_n·y_n)
/// ```
///
/// `(cx, cy)` is deliberately NOT pulled toward the image center: on
/// multi-detector mosaics (TESS, Kepler, Rubin, …) the camera's optical
/// axis lies far off any single detector's center — near a corner for a
/// TESS CCD — and that is where the radial distortion is physically
/// centered. Only a tiny tie-breaking prior on `(cx, cy, p1, p2)`
/// (negligible against any real signal) picks the centered representative
/// when the data cannot distinguish — e.g. a distortion-free pinhole,
/// where the optical center is unidentifiable (a center shift `δc·k1` is
/// first-order indistinguishable from a tangential `p2` term).
///
/// LM is delegated to [`numeris::optim::least_squares_lm_dyn`]. The outer
/// loop performs MAD-based sigma-clipping: after each LM convergence,
/// re-mask inliers based on the residual distribution and re-call LM.
///
/// Warm-starts `(k1, k2, k3)` from a non-centered linear fit
/// ([`fit_radial_ls`]); `γ` starts at 1 and `(cx, cy, p1, p2)` at 0.
pub(super) fn fit_radial_centered_sigma_clip(
    points: &[MatchedPoint],
    config: &DistortionFitConfig,
) -> CenteredRadialFitResult {
    // Normalize coordinates so the field radius is ~1. Without this the
    // Jacobian columns span ~20 orders of magnitude (the γ column is O(r),
    // the k3 column O(r⁷)) and LM hard-fails or converges to wrong local
    // minima at real-camera magnitudes. Dimensionless parameters: ĉ = c/L,
    // k̂ᵢ = kᵢ·L^2i, p̂ = p·L; γ is already dimensionless.
    let norm = points
        .iter()
        .map(|p| p.x_ideal.hypot(p.y_ideal))
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let npoints: Vec<MatchedPoint> = points
        .iter()
        .map(|p| MatchedPoint {
            x_obs: p.x_obs / norm,
            y_obs: p.y_obs / norm,
            x_ideal: p.x_ideal / norm,
            y_ideal: p.y_ideal / norm,
        })
        .collect();

    // Warm-start `k`s from a quick non-centered linear fit.
    let initial_mask = vec![true; npoints.len()];
    let (k1_init, k2_init, k3_init) = fit_radial_ls(&npoints, &initial_mask);
    let mut x = DynVector::<f64>::from_vec(vec![
        0.0,     // cx
        0.0,     // cy
        1.0,     // gamma
        k1_init, // k1
        k2_init, // k2
        k3_init, // k3
        0.0,     // p1
        0.0,     // p2
    ]);
    let mut mask = initial_mask;
    let mut total_lm_iters = 0u32;

    // Outer sigma-clip iterations
    for _outer in 0..config.max_iterations {
        let n_inliers = mask.iter().filter(|&&m| m).count();
        if n_inliers < 8 {
            break;
        }
        let prev_x = x.clone();

        match run_intrinsics_lm(&npoints, &mask, &x) {
            Ok((new_x, iters)) => {
                x = new_x;
                total_lm_iters += iters;
            }
            Err(()) => break,
        }

        // Sigma-clip on residuals at current params.
        let residuals = intrinsics_residuals(&npoints, x.as_slice());
        let inlier_resids: Vec<f64> = residuals
            .iter()
            .zip(&mask)
            .filter(|(_, &m)| m)
            .map(|(&r, _)| r)
            .collect();
        if inlier_resids.is_empty() {
            break;
        }
        let sigma = mad_sigma(&inlier_resids);
        if sigma < 1e-12 / norm {
            break;
        }
        let threshold = config.sigma_clip * sigma;
        let new_mask: Vec<bool> = residuals.iter().map(|&r| r <= threshold).collect();
        let mask_changed = mask.iter().zip(&new_mask).any(|(&a, &b)| a != b);
        mask = new_mask;
        let params_changed = (0..8).any(|i| (x[i] - prev_x[i]).abs() > 1e-12);
        if !mask_changed && !params_changed {
            break;
        }
        if mask.iter().filter(|&&m| m).count() < 8 {
            break;
        }
    }

    // Stage 2: optional fixed-pixel-threshold recovery — pull back inliers
    // whose Euclidean residual is below `stage2_threshold_px` regardless of
    // sigma-clip rejection. Refit once if any new inliers join.
    if let Some(threshold_px) = config.stage2_threshold_px {
        let residuals = intrinsics_residuals(&npoints, x.as_slice());
        let threshold = threshold_px / norm;
        let mask_s2: Vec<bool> = residuals.iter().map(|&r| r <= threshold).collect();
        let n_recovered = mask_s2
            .iter()
            .zip(&mask)
            .filter(|(&s2, &s1)| s2 && !s1)
            .count();
        if n_recovered > 0 && mask_s2.iter().filter(|&&m| m).count() >= 8 {
            mask = mask_s2;
            if let Ok((new_x, iters)) = run_intrinsics_lm(&npoints, &mask, &x) {
                x = new_x;
                total_lm_iters += iters;
            }
        }
    }

    // Denormalize back to pixel units.
    let n2 = norm * norm;
    CenteredRadialFitResult {
        cx: x[0] * norm,
        cy: x[1] * norm,
        gamma: x[2],
        k1: x[3] / n2,
        k2: x[4] / (n2 * n2),
        k3: x[5] / (n2 * n2 * n2),
        p1: x[6] / norm,
        p2: x[7] / norm,
        mask,
        iterations: total_lm_iters,
    }
}

/// Predicted observed position under the intrinsics model, for one point.
/// `params` = `[cx, cy, gamma, k1, k2, k3, p1, p2]` (all in the same units
/// as the point coordinates).
fn intrinsics_predict(p: &MatchedPoint, params: &[f64]) -> (f64, f64) {
    let (cx, cy, gamma) = (params[0], params[1], params[2]);
    let (k1, k2, k3, p1, p2) = (params[3], params[4], params[5], params[6], params[7]);
    let xn = p.x_ideal - cx;
    let yn = p.y_ideal - cy;
    let r2 = xn * xn + yn * yn;
    let r4 = r2 * r2;
    let r6 = r2 * r4;
    let rad = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
    let dx = xn * rad + 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn);
    let dy = yn * rad + p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn;
    (cx + gamma * dx, cy + gamma * dy)
}

/// Per-point Euclidean residual under the intrinsics model.
/// `params` as in [`intrinsics_predict`].
pub(super) fn intrinsics_residuals(points: &[MatchedPoint], params: &[f64]) -> Vec<f64> {
    points
        .iter()
        .map(|p| {
            let (px, py) = intrinsics_predict(p, params);
            (p.x_obs - px).hypot(p.y_obs - py)
        })
        .collect()
}

/// Single Levenberg-Marquardt run on the intrinsics model, dispatched to
/// [`numeris::optim::least_squares_lm_dyn`].
///
/// The residual vector has length `2·N_inliers + 4`: two rows per inlier
/// point (x and y) plus four tie-breaking rows `√μ·(cx, cy, p1, p2)`. The
/// tie-break weight is far below any real data term; it only selects the
/// centered, radial-only representative when the data leaves the
/// `(cx, cy) ↔ (p1, p2)` ridge exactly degenerate (see
/// [`fit_radial_centered_sigma_clip`]). Returns `Err(())` if there aren't
/// enough inliers or if LM fails.
fn run_intrinsics_lm(
    points: &[MatchedPoint],
    mask: &[bool],
    x0: &DynVector<f64>,
) -> Result<(DynVector<f64>, u32), ()> {
    let inlier_indices: Vec<usize> = mask
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| if m { Some(i) } else { None })
        .collect();
    if inlier_indices.len() < 8 {
        return Err(());
    }
    // Tie-break weight √μ (normalized coordinates): cost μ·ĉ² ≤ 1e-8 even
    // at ĉ ~ 1 — at least three orders below the smallest realistic data
    // term, so it never fights real signal.
    const SQRT_MU_TIE: f64 = 1e-4;
    let m = 2 * inlier_indices.len() + 4;

    // Closures borrow the inlier index list and points. The residual
    // function returns a column vector of length m. The Jacobian returns
    // an m×8 matrix.
    let residual = |x: &DynVector<f64>| -> DynVector<f64> {
        let mut r = DynVector::<f64>::zeros(m);
        for (row_pair, &i) in inlier_indices.iter().enumerate() {
            let p = &points[i];
            let (px, py) = intrinsics_predict(p, x.as_slice());
            r[2 * row_pair] = p.x_obs - px;
            r[2 * row_pair + 1] = p.y_obs - py;
        }
        // Tie-break rows: cost contribution (√μ·cx)² = μ·cx², etc.
        r[m - 4] = SQRT_MU_TIE * x[0];
        r[m - 3] = SQRT_MU_TIE * x[1];
        r[m - 2] = SQRT_MU_TIE * x[6];
        r[m - 1] = SQRT_MU_TIE * x[7];
        r
    };

    let jacobian = |x: &DynVector<f64>| -> DynMatrix<f64> {
        // Numeris LM uses the convention r(x) is the residual; gradient = Jᵀr.
        // R = obs − predicted, so rows are ∂R/∂params = −∂predicted/∂params.
        let cx = x[0];
        let cy = x[1];
        let gamma = x[2];
        let k1 = x[3];
        let k2 = x[4];
        let k3 = x[5];
        let p1 = x[6];
        let p2 = x[7];
        let mut j = DynMatrix::<f64>::zeros(m, 8);
        for (row_pair, &i) in inlier_indices.iter().enumerate() {
            let p = &points[i];
            let xn = p.x_ideal - cx;
            let yn = p.y_ideal - cy;
            let r2 = xn * xn + yn * yn;
            let r4 = r2 * r2;
            let r6 = r2 * r4;
            let rad = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
            let radp = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4; // d(rad)/d(r²)
            let dx = xn * rad + 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn);
            let dy = yn * rad + p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn;
            // ∂(distorted)/∂xn and cross terms, shared between rows.
            let ddx_dxn = rad + 2.0 * xn * xn * radp + 2.0 * p1 * yn + 6.0 * p2 * xn;
            let ddx_dyn = 2.0 * xn * yn * radp + 2.0 * p1 * xn + 2.0 * p2 * yn;
            let ddy_dyn = rad + 2.0 * yn * yn * radp + 6.0 * p1 * yn + 2.0 * p2 * xn;
            let ddy_dxn = ddx_dyn; // symmetric mixed term
            let row_x = 2 * row_pair;
            let row_y = row_x + 1;
            // x equation: R = x_obs − cx − γ·dx, with ∂xn/∂cx = −1.
            j[(row_x, 0)] = -1.0 + gamma * ddx_dxn;
            j[(row_x, 1)] = gamma * ddx_dyn;
            j[(row_x, 2)] = -dx;
            j[(row_x, 3)] = -gamma * xn * r2;
            j[(row_x, 4)] = -gamma * xn * r4;
            j[(row_x, 5)] = -gamma * xn * r6;
            j[(row_x, 6)] = -gamma * 2.0 * xn * yn;
            j[(row_x, 7)] = -gamma * (r2 + 2.0 * xn * xn);
            // y equation
            j[(row_y, 0)] = gamma * ddy_dxn;
            j[(row_y, 1)] = -1.0 + gamma * ddy_dyn;
            j[(row_y, 2)] = -dy;
            j[(row_y, 3)] = -gamma * yn * r2;
            j[(row_y, 4)] = -gamma * yn * r4;
            j[(row_y, 5)] = -gamma * yn * r6;
            j[(row_y, 6)] = -gamma * (r2 + 2.0 * yn * yn);
            j[(row_y, 7)] = -gamma * 2.0 * xn * yn;
        }
        // Tie-break rows: ∂(√μ·cx)/∂cx = √μ, etc.
        j[(m - 4, 0)] = SQRT_MU_TIE;
        j[(m - 3, 1)] = SQRT_MU_TIE;
        j[(m - 2, 6)] = SQRT_MU_TIE;
        j[(m - 1, 7)] = SQRT_MU_TIE;
        j
    };

    // Coordinates are normalized (field radius ~1, residuals ~1e-4), so the
    // relative tolerances are scale-free. The joint (cx, cy, γ, p1, p2) fit
    // has a long shallow valley; give LM room to walk it — numeris returns
    // a hard error (discarding the iterate) when max_iter is hit.
    let settings = LmSettings::<f64> {
        max_iter: 500,
        f_tol: 1e-11,
        x_tol: 1e-11,
        ..LmSettings::default()
    };
    let result = least_squares_lm_dyn(residual, jacobian, x0, &settings).map_err(|_| ())?;
    Ok((result.x, result.iterations as u32))
}

/// Root-mean-square of the masked-in per-point residuals.
pub(super) fn masked_rms(residuals: &[f64], mask: &[bool]) -> f64 {
    let mut sum_sq = 0.0_f64;
    let mut n = 0usize;
    for (&r, &m) in residuals.iter().zip(mask) {
        if m {
            sum_sq += r * r;
            n += 1;
        }
    }
    if n == 0 {
        0.0
    } else {
        (sum_sq / n as f64).sqrt()
    }
}

/// Like [`compute_corrected_rmse`] but applies an optional crpix shift
/// before the distortion model evaluates. Used for centered radial where
/// the model is anchored at `(cx, cy)` rather than the geometric origin.
pub(super) fn compute_corrected_rmse_centered(
    points: &[MatchedPoint],
    mask: &[bool],
    distortion: &Distortion,
    crpix: [f64; 2],
) -> f64 {
    if points.is_empty() {
        return 0.0;
    }
    let mut sum_sq = 0.0_f64;
    let mut n = 0usize;
    for (i, p) in points.iter().enumerate() {
        if !mask[i] {
            continue;
        }
        // Forward model: ideal → observed in the geometric frame.
        // For centered models, shift to the optical-axis frame, distort,
        // shift back.
        let (predicted_x, predicted_y) = match distortion {
            Distortion::None => (p.x_ideal, p.y_ideal),
            _ => {
                let (dx, dy) = distortion.distort(p.x_ideal - crpix[0], p.y_ideal - crpix[1]);
                (dx + crpix[0], dy + crpix[1])
            }
        };
        let rx = p.x_obs - predicted_x;
        let ry = p.y_obs - predicted_y;
        sum_sq += rx * rx + ry * ry;
        n += 1;
    }
    if n == 0 {
        0.0
    } else {
        (sum_sq / n as f64).sqrt()
    }
}

/// Solve `(k1, k2, k3)` from matched points using least squares.
///
/// Model: `x_obs - x_ideal = x_ideal · (k1·r² + k2·r⁴ + k3·r⁶)`
///        `y_obs - y_ideal = y_ideal · (k1·r² + k2·r⁴ + k3·r⁶)`
///
/// Stacks both x and y equations into one system with 3 unknowns.
fn fit_radial_ls(points: &[MatchedPoint], mask: &[bool]) -> (f64, f64, f64) {
    let inlier_count: usize = mask.iter().filter(|&&m| m).count();

    if inlier_count < 3 {
        return (0.0, 0.0, 0.0);
    }

    let nrows = inlier_count * 2;
    let mut a_mat = DynMatrix::<f64>::zeros(nrows, 3);
    let mut b_vec = DynVector::<f64>::zeros(nrows);

    let mut row = 0;
    for (i, p) in points.iter().enumerate() {
        if !mask[i] {
            continue;
        }
        let r2 = p.x_ideal * p.x_ideal + p.y_ideal * p.y_ideal;
        let r4 = r2 * r2;
        let r6 = r2 * r4;

        a_mat[(row, 0)] = p.x_ideal * r2;
        a_mat[(row, 1)] = p.x_ideal * r4;
        a_mat[(row, 2)] = p.x_ideal * r6;
        b_vec[row] = p.x_obs - p.x_ideal;
        row += 1;

        a_mat[(row, 0)] = p.y_ideal * r2;
        a_mat[(row, 1)] = p.y_ideal * r4;
        a_mat[(row, 2)] = p.y_ideal * r6;
        b_vec[row] = p.y_obs - p.y_ideal;
        row += 1;
    }

    let coeffs = a_mat
        .solve_qr(&b_vec)
        .unwrap_or_else(|_| DynVector::zeros(3));

    (coeffs[0], coeffs[1], coeffs[2])
}

// ── Polynomial (SIP-like) distortion fitting ────────────────────────────────

/// Result of a sigma-clipped polynomial fit (forward + inverse).
pub(super) struct PolyFitResult {
    pub a_coeffs: Vec<f64>,
    pub b_coeffs: Vec<f64>,
    pub ap_coeffs: Vec<f64>,
    pub bp_coeffs: Vec<f64>,
    pub mask: Vec<bool>,
    pub iterations: u32,
}

/// Per-point radial residual (in pixels) of the forward SIP polynomial model.
///
/// For each matched point: evaluate the forward polynomial at the normalized
/// ideal coordinates `(u, v) = (x_ideal, y_ideal) / scale`, then return the
/// Euclidean distance between the observed offset and the modeled offset.
fn poly_point_residuals(
    points: &[MatchedPoint],
    pairs: &[(u32, u32)],
    scale: f64,
    a_coeffs: &[f64],
    b_coeffs: &[f64],
) -> Vec<f64> {
    points
        .iter()
        .map(|p| {
            let u = p.x_ideal / scale;
            let v = p.y_ideal / scale;
            let dx_model: f64 = pairs
                .iter()
                .enumerate()
                .map(|(i, &(pp, qq))| a_coeffs[i] * u.powi(pp as i32) * v.powi(qq as i32))
                .sum();
            let dy_model: f64 = pairs
                .iter()
                .enumerate()
                .map(|(i, &(pp, qq))| b_coeffs[i] * u.powi(pp as i32) * v.powi(qq as i32))
                .sum();
            let rx = p.x_obs - p.x_ideal - dx_model * scale;
            let ry = p.y_obs - p.y_ideal - dy_model * scale;
            (rx * rx + ry * ry).sqrt()
        })
        .collect()
}

/// Fit a polynomial distortion model with iterative sigma-clipping.
///
/// This is the core fitting loop extracted for reuse by multi-image calibration.
/// It performs:
/// 1. Initial forward polynomial LS fit.
/// 2. Iterative MAD-based sigma-clipping.
/// 3. Stage 2 outlier recovery (optional, controlled by `config.stage2_threshold_px`).
/// 4. Inverse polynomial fit on the final inlier set.
///
/// `points` are matched observations, `order` is the polynomial order,
/// `scale` is the normalization factor (typically image_width / 2).
pub(super) fn fit_polynomial_sigma_clip(
    points: &[MatchedPoint],
    order: u32,
    scale: f64,
    config: &DistortionFitConfig,
) -> PolyFitResult {
    let n = points.len();
    let ncoeffs = num_coeffs(order);
    let pairs = term_pairs(order);

    let mut mask = vec![true; n];
    let mut iterations = 0u32;
    let mut a_coeffs = vec![0.0; ncoeffs];
    let mut b_coeffs = vec![0.0; ncoeffs];

    // Initial fit
    fit_poly_ls(points, &mask, &pairs, scale, &mut a_coeffs, &mut b_coeffs);

    for iter in 0..config.max_iterations {
        iterations = iter + 1;

        // Compute residuals using current model
        let residuals = poly_point_residuals(points, &pairs, scale, &a_coeffs, &b_coeffs);

        // MAD-based robust clipping
        let inlier_resids: Vec<f64> = residuals
            .iter()
            .zip(&mask)
            .filter(|(_, &m)| m)
            .map(|(&r, _)| r)
            .collect();

        if inlier_resids.is_empty() {
            break;
        }

        let sigma = mad_sigma(&inlier_resids);

        if sigma < 1e-12 {
            break;
        }

        let threshold = config.sigma_clip * sigma;
        let new_mask: Vec<bool> = residuals.iter().map(|&r| r <= threshold).collect();

        let changed = mask.iter().zip(&new_mask).any(|(&a, &b)| a != b);
        mask = new_mask;

        if !changed {
            break;
        }

        let n_inliers = mask.iter().filter(|&&m| m).count();
        if n_inliers < ncoeffs {
            debug!(
                "Too few inliers ({}) for polynomial fit after sigma-clip",
                n_inliers
            );
            break;
        }

        fit_poly_ls(points, &mask, &pairs, scale, &mut a_coeffs, &mut b_coeffs);
    }

    // Stage 2: recover outliers below a threshold
    if let Some(threshold_px) = config.stage2_threshold_px {
        let residuals = poly_point_residuals(points, &pairs, scale, &a_coeffs, &b_coeffs);

        let mask_s2: Vec<bool> = residuals.iter().map(|&r| r <= threshold_px).collect();
        let n_recovered = mask_s2
            .iter()
            .zip(&mask)
            .filter(|(&s2, &s1)| s2 && !s1)
            .count();

        if n_recovered > 0 {
            mask = mask_s2;
            let n_inliers = mask.iter().filter(|&&m| m).count();
            if n_inliers >= ncoeffs {
                fit_poly_ls(points, &mask, &pairs, scale, &mut a_coeffs, &mut b_coeffs);
            }
        }
    }

    // The inverse polynomial (distorted → ideal) is no longer fit:
    // PolynomialDistortion::undistort uses Newton iteration on the forward
    // polynomial, which is exact (limited only by forward expressiveness).
    // The ap/bp fields remain in PolynomialDistortion for binary format
    // compatibility but are zero-valued.
    let ap_coeffs = vec![0.0; ncoeffs];
    let bp_coeffs = vec![0.0; ncoeffs];

    PolyFitResult {
        a_coeffs,
        b_coeffs,
        ap_coeffs,
        bp_coeffs,
        mask,
        iterations,
    }
}

/// Fit a polynomial (SIP-like) distortion model from plate-solve results.
///
/// This model fits arbitrary 2D polynomial correction terms:
///   x_obs = x_ideal + Σ A_pq · (x_ideal/s)^p · (y_ideal/s)^q   (s = scale, 0 ≤ p+q ≤ order)
///   y_obs = y_ideal + Σ B_pq · (x_ideal/s)^p · (y_ideal/s)^q
/// for 2 ≤ p+q ≤ order.
///
/// An inverse polynomial (distorted → ideal) is also fitted for efficient
/// undistortion in the solver.
///
/// The `order` parameter determines the polynomial complexity:
/// - order 2: 6 terms per axis (12 total) — offset + linear + quadratic
/// - order 3: 10 terms per axis (20 total) — + cubic
/// - order 4: 15 terms per axis (30 total) — + quartic (recommended for TESS)
/// - order 5: 21 terms per axis (42 total) — + quintic
///
/// Order-0 terms absorb optical center offset; order-1 terms absorb residual
/// plate scale and rotation errors.
pub fn fit_polynomial_distortion(
    solve_results: &[&SolveResult],
    centroids: &[&[Centroid]],
    database: &SolverDatabase,
    image_width: u32,
    order: u32,
    config: &DistortionFitConfig,
) -> DistortionFitResult {
    assert_eq!(
        solve_results.len(),
        centroids.len(),
        "solve_results and centroids must have the same length"
    );
    assert!(
        (2..=6).contains(&order),
        "polynomial order must be in [2, 6]"
    );

    let id_to_idx = build_id_lookup(database);
    let points = gather_matched_points(solve_results, centroids, database, &id_to_idx, image_width);

    if points.is_empty() {
        return DistortionFitResult {
            model: Distortion::None,
            crpix: None,
            focal_scale: 1.0,
            rmse_before_px: 0.0,
            rmse_after_px: 0.0,
            n_inliers: 0,
            n_outliers: 0,
            iterations: 0,
        };
    }

    let n = points.len();
    let ncoeffs = num_coeffs(order);
    let scale = image_width as f64 / 2.0;

    // Minimum data points: need at least ncoeffs matched pairs
    if n < ncoeffs {
        let rmse_raw = compute_rmse_px(&points);
        debug!(
            "Too few matched points ({}) for order-{} polynomial fit ({} coefficients needed)",
            n, order, ncoeffs
        );
        return DistortionFitResult {
            model: Distortion::None,
            crpix: None,
            focal_scale: 1.0,
            rmse_before_px: rmse_raw,
            rmse_after_px: rmse_raw,
            n_inliers: n,
            n_outliers: 0,
            iterations: 0,
        };
    }

    // Delegate to the reusable sigma-clip helper
    let fit = fit_polynomial_sigma_clip(&points, order, scale, config);

    let model = PolynomialDistortion::new(
        order,
        scale,
        fit.a_coeffs,
        fit.b_coeffs,
        fit.ap_coeffs,
        fit.bp_coeffs,
    );
    let dist = Distortion::Polynomial(model.clone());
    // Compute before/after RMSE on the SAME inlier set for a fair comparison
    let rmse_before = compute_corrected_rmse(&points, &fit.mask, &Distortion::None);
    let rmse_after = compute_corrected_rmse(&points, &fit.mask, &dist);
    let n_inliers = fit.mask.iter().filter(|&&m| m).count();

    debug!(
        "Polynomial (order {}) fit: {} coefficients/axis, inliers={}/{}, RMSE {:.3} → {:.3} px",
        order, ncoeffs, n_inliers, n, rmse_before, rmse_after
    );

    DistortionFitResult {
        model: dist,
        crpix: None,
        focal_scale: 1.0,
        rmse_before_px: rmse_before,
        rmse_after_px: rmse_after,
        n_inliers,
        n_outliers: n - n_inliers,
        iterations: fit.iterations,
    }
}

// ── Internal helpers ────────────────────────────────────────────────────────

/// Build a HashMap from catalog_id → index into star_vectors.
pub(super) fn build_id_lookup(database: &SolverDatabase) -> HashMap<i64, usize> {
    database
        .star_catalog_ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect()
}

/// Gather all matched points across multiple solve results.
///
/// For each matched pair, project the catalog star to pixel coordinates using
/// the solve result's rotation and FOV, and pair it with the observed centroid.
fn gather_matched_points(
    solve_results: &[&SolveResult],
    centroids: &[&[Centroid]],
    database: &SolverDatabase,
    id_to_idx: &HashMap<i64, usize>,
    image_width: u32,
) -> Vec<MatchedPoint> {
    let mut points = Vec::new();

    for (sr, cents) in solve_results.iter().zip(centroids.iter()) {
        let Ok(sr) = sr else {
            continue; // skip failed solves
        };

        // True pinhole pixel scale (1/f).
        let pixel_scale = {
            let f = (image_width as f32 / 2.0) / (sr.fov_rad / 2.0).tan();
            1.0 / f
        };
        let rot: Matrix3<f32> = sr.qicrs2cam.to_rotation_matrix();

        let parity_sign: f64 = if sr.parity_flip { -1.0 } else { 1.0 };

        for (match_idx, &cat_id) in sr.matched_catalog_ids.iter().enumerate() {
            let cent_idx = sr.matched_centroid_indices[match_idx];
            if cent_idx >= cents.len() {
                continue;
            }

            let star_idx = match id_to_idx.get(&cat_id) {
                Some(&idx) => idx,
                None => continue,
            };

            let sv = &database.star_vectors[star_idx];
            let x_obs = cents[cent_idx].x as f64;
            let y_obs = cents[cent_idx].y as f64;
            if let Some(mp) =
                project_to_matched_point(rot, sv, parity_sign, pixel_scale as f64, x_obs, y_obs)
            {
                points.push(mp);
            }
        }
    }

    points
}

/// Project a catalog star (ICRS unit vector `sv`) through `rot` into ideal
/// pinhole pixel coordinates and pair it with an observed centroid `(x_obs,
/// y_obs)`.
///
/// Returns `None` when the star projects behind the camera (`cam_z ≤ 0`).
/// `parity_sign` is `-1.0` when the image x-axis is flipped; `pixel_scale` is
/// radians per pixel (`1/focal_length_px`). Shared by the single-image gather
/// and the multi-image calibration Phase 2.
pub(super) fn project_to_matched_point(
    rot: Matrix3<f32>,
    sv: &[f32; 3],
    parity_sign: f64,
    pixel_scale: f64,
    x_obs: f64,
    y_obs: f64,
) -> Option<MatchedPoint> {
    let icrs_v = numeris::Vector3::from_array([sv[0], sv[1], sv[2]]);
    let cam_v = rot * icrs_v;
    if cam_v[2] <= 0.0 {
        return None;
    }
    let x_ideal = parity_sign * (cam_v[0] as f64) / (cam_v[2] as f64) / pixel_scale;
    let y_ideal = (cam_v[1] as f64) / (cam_v[2] as f64) / pixel_scale;
    Some(MatchedPoint {
        x_obs,
        y_obs,
        x_ideal,
        y_ideal,
    })
}

/// Compute RMS pixel residual (uncorrected) across all points.
fn compute_rmse_px(points: &[MatchedPoint]) -> f64 {
    if points.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = points
        .iter()
        .map(|p| {
            let dx = p.x_obs - p.x_ideal;
            let dy = p.y_obs - p.y_ideal;
            dx * dx + dy * dy
        })
        .sum();
    (sum_sq / points.len() as f64).sqrt()
}

/// Compute RMS pixel residual after applying distortion correction to inliers.
pub(super) fn compute_corrected_rmse(
    points: &[MatchedPoint],
    mask: &[bool],
    distortion: &Distortion,
) -> f64 {
    let mut sum_sq = 0.0;
    let mut count = 0;

    for (i, p) in points.iter().enumerate() {
        if !mask[i] {
            continue;
        }
        // Undistort the observed position, then compare to ideal
        let (xu, yu) = distortion.undistort(p.x_obs, p.y_obs);
        let dx = xu - p.x_ideal;
        let dy = yu - p.y_ideal;
        sum_sq += dx * dx + dy * dy;
        count += 1;
    }

    if count == 0 {
        return 0.0;
    }
    (sum_sq / count as f64).sqrt()
}

/// Compute the percentile of a sorted slice. `p` is in [0, 1].
pub(super) fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let mut values = sorted.to_vec();
    values.sort_by(f64::total_cmp);
    let idx = (p * (values.len() - 1) as f64).round() as usize;
    values[idx.min(values.len() - 1)]
}

/// MAD-derived σ estimate (`1.4826 · MAD`) of a residual slice.
///
/// Uses the [`percentile`] helper (which sorts internally) for both the median
/// and the median-absolute-deviation, matching the legacy fit convention.
fn mad_sigma(inlier_resids: &[f64]) -> f64 {
    let median = percentile(inlier_resids, 0.5);
    let abs_devs: Vec<f64> = inlier_resids.iter().map(|&r| (r - median).abs()).collect();
    let mad = percentile(&abs_devs, 0.5);
    mad * 1.4826
}

/// Fit the forward polynomial (ideal → distorted) by least-squares.
///
/// Model: x_obs = x_ideal + Σ A_pq · u^p · v^q   (u = x_ideal/scale, v = y_ideal/scale, 0 ≤ p+q ≤ order)
/// Stacks x and y equations, solves each axis independently.
pub(super) fn fit_poly_ls(
    points: &[MatchedPoint],
    mask: &[bool],
    pairs: &[(u32, u32)],
    scale: f64,
    a_coeffs: &mut [f64],
    b_coeffs: &mut [f64],
) {
    let ncoeffs = pairs.len();
    let n_inliers: usize = mask.iter().filter(|&&m| m).count();
    if n_inliers < ncoeffs {
        return;
    }

    let mut a_mat = DynMatrix::<f64>::zeros(n_inliers, ncoeffs);
    let mut bx_vec = DynVector::<f64>::zeros(n_inliers);
    let mut by_vec = DynVector::<f64>::zeros(n_inliers);

    let mut row = 0;
    for (i, p) in points.iter().enumerate() {
        if !mask[i] {
            continue;
        }
        let u = p.x_ideal / scale;
        let v = p.y_ideal / scale;

        for (j, &(pp, qq)) in pairs.iter().enumerate() {
            a_mat[(row, j)] = u.powi(pp as i32) * v.powi(qq as i32);
        }
        bx_vec[row] = (p.x_obs - p.x_ideal) / scale;
        by_vec[row] = (p.y_obs - p.y_ideal) / scale;
        row += 1;
    }

    // A rank-deficient design matrix (e.g. collinear or too-few points) leaves
    // the coefficients at their previous values; log it so a silently-degraded
    // fit isn't reported as a normal one.
    // A rank-deficient design matrix (e.g. collinear or too-few points) leaves
    // the coefficients at their previous values; log it so a silently-degraded
    // fit isn't reported as a normal one.
    match a_mat.solve_qr(&bx_vec) {
        Ok(cx) => {
            for j in 0..ncoeffs {
                a_coeffs[j] = cx[j];
            }
        }
        Err(_) => debug!("fit_poly_ls: x-axis QR solve failed; keeping prior coeffs"),
    }

    match a_mat.solve_qr(&by_vec) {
        Ok(cy) => {
            for j in 0..ncoeffs {
                b_coeffs[j] = cy[j];
            }
        }
        Err(_) => debug!("fit_poly_ls: y-axis QR solve failed; keeping prior coeffs"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test that fitting recovers known radial distortion from synthetic data.
    #[test]
    fn test_fit_radial_synthetic() {
        let true_k1 = -7e-9;
        let true_k2 = 2e-15;
        let true_k3 = 0.0;
        let true_distortion = RadialDistortion::new(true_k1, true_k2, true_k3);

        let mut points = Vec::new();
        for ix in -5..=5 {
            for iy in -5..=5 {
                let x_ideal = ix as f64 * 100.0;
                let y_ideal = iy as f64 * 100.0;
                let (x_obs, y_obs) = true_distortion.distort(x_ideal, y_ideal);
                points.push(MatchedPoint {
                    x_obs,
                    y_obs,
                    x_ideal,
                    y_ideal,
                });
            }
        }

        let mask = vec![true; points.len()];
        let (k1, k2, k3) = fit_radial_ls(&points, &mask);

        assert!(
            (k1 - true_k1).abs() < 1e-12,
            "k1: fitted={:.6e}, true={:.6e}",
            k1,
            true_k1,
        );
        assert!(
            (k2 - true_k2).abs() < 1e-18,
            "k2: fitted={:.6e}, true={:.6e}",
            k2,
            true_k2,
        );
        assert!(k3.abs() < 1e-18, "k3: fitted={:.3e}, expected ~0", k3);
    }

    /// The joint fit must separate a pure focal-scale error (wrong anchor
    /// focal length) from lens distortion: `gamma` recovers the scale and
    /// the rescaled Brown-Conrady coefficients stay ~0.
    #[test]
    fn test_fit_radial_recovers_focal_scale() {
        let true_gamma = 1.009; // ~1% focal-length error, as seen on TESS

        let mut points = Vec::new();
        for ix in -5..=5 {
            for iy in -5..=5 {
                let x_ideal = ix as f64 * 100.0;
                let y_ideal = iy as f64 * 100.0;
                points.push(MatchedPoint {
                    x_obs: x_ideal * true_gamma,
                    y_obs: y_ideal * true_gamma,
                    x_ideal,
                    y_ideal,
                });
            }
        }

        let config = DistortionFitConfig::default();
        let fit = fit_radial_centered_sigma_clip(&points, &config);

        assert!(
            (fit.gamma - true_gamma).abs() < 1e-6,
            "gamma: fitted={:.8}, true={:.8}",
            fit.gamma,
            true_gamma,
        );
        let model = fit.rescaled_model();
        // Residual lens distortion in the rescaled model must be far below
        // a centroid width well into the field.
        let (xd, yd) = model.distort(500.0, 500.0);
        assert!(
            (xd - 500.0).abs() < 1e-3 && (yd - 500.0).abs() < 1e-3,
            "rescaled model not ~identity: distort(500, 500) = ({xd}, {yd})",
        );
    }

    /// Focal scale and genuine radial distortion fit jointly without
    /// trading off.
    #[test]
    fn test_fit_radial_scale_and_distortion_jointly() {
        let true_gamma = 0.995;
        let true_k1 = -7e-9;
        let true_distortion = RadialDistortion::new(true_k1, 0.0, 0.0);

        let mut points = Vec::new();
        for ix in -7..=7 {
            for iy in -7..=7 {
                let x_ideal = ix as f64 * 100.0;
                let y_ideal = iy as f64 * 100.0;
                let (xd, yd) = true_distortion.distort(x_ideal, y_ideal);
                points.push(MatchedPoint {
                    x_obs: xd * true_gamma,
                    y_obs: yd * true_gamma,
                    x_ideal,
                    y_ideal,
                });
            }
        }

        let config = DistortionFitConfig::default();
        let fit = fit_radial_centered_sigma_clip(&points, &config);

        assert!(
            (fit.gamma - true_gamma).abs() < 1e-5,
            "gamma: fitted={:.8}, true={:.8}",
            fit.gamma,
            true_gamma,
        );
        assert!(
            (fit.k1 - true_k1).abs() < 1e-11,
            "k1: fitted={:.6e}, true={:.6e}",
            fit.k1,
            true_k1,
        );
    }

    /// Mosaic-camera geometry: the optical axis (distortion center) sits
    /// near a CCD corner, ~1.5 field-radii from the image center — the TESS
    /// situation. The unregularized center must travel there, and the joint
    /// fit must still recover scale and distortion.
    #[test]
    fn test_fit_radial_mosaic_corner_center() {
        let true_gamma = 1.009;
        let (true_cx, true_cy) = (-1100.0, -1080.0);
        let true_d = RadialDistortion::with_tangential(-5e-9, 1e-15, -1e-21, 1e-7, -2e-7);

        let mut points = Vec::new();
        let mut k = 0u32;
        for ix in -31..=31 {
            for iy in -31..=31 {
                let x_ideal = ix as f64 * 33.0 + (k % 17) as f64; // break grid symmetry
                let y_ideal = iy as f64 * 33.0 + (k % 13) as f64;
                k += 1;
                let (dx, dy) = true_d.distort(x_ideal - true_cx, y_ideal - true_cy);
                points.push(MatchedPoint {
                    x_obs: true_cx + true_gamma * dx,
                    y_obs: true_cy + true_gamma * dy,
                    x_ideal,
                    y_ideal,
                });
            }
        }

        let config = DistortionFitConfig::default();
        let fit = fit_radial_centered_sigma_clip(&points, &config);

        let resid = intrinsics_residuals(
            &points,
            &[
                fit.cx, fit.cy, fit.gamma, fit.k1, fit.k2, fit.k3, fit.p1, fit.p2,
            ],
        );
        let rms = masked_rms(&resid, &fit.mask);
        assert!(rms < 0.01, "rms {rms:.4} px on noiseless synthetic data");
        assert!(
            (fit.gamma - true_gamma).abs() < 1e-4,
            "gamma: fitted={:.8}, true={:.8}",
            fit.gamma,
            true_gamma,
        );
        assert!(
            (fit.cx - true_cx).abs() < 20.0 && (fit.cy - true_cy).abs() < 20.0,
            "center: fitted=({:.1}, {:.1}), true=({true_cx}, {true_cy})",
            fit.cx,
            fit.cy,
        );
    }
}
