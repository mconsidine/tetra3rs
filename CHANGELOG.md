# Changelog

## 0.9.0

### Upgrading from 0.8

The short list — full details in the sections referenced below.

- **CCL-path `sigma_threshold` now means true Gaussian sigmas** (the old
  estimator ran ~40% low). Multiply configured values by ≈0.6 (e.g. `5.0` →
  `3.0`) to keep your previous effective detection depth;
  `FastCentroidConfig` is unaffected. See *Fixed — CCL-path noise estimator*.
- **The matched filter is now on by default** (`matched_filter_sigma =
  Some(1.5)`), with the detection threshold auto-compensated for the kernel's
  noise suppression — no threshold retuning needed. Set `None` to opt out.
  See *Changed — matched filter on by default*.
- **`max_sharpness` defaults to `0.9`** — a hot-pixel / cosmic-ray gate that
  passes any PSF spanning multiple pixels. Set `None` for severely
  undersampled data (PSF FWHM below ~1.5 px, e.g. resampled survey cutouts).
  See *Added — extraction quality gates*.
- **`match_threshold` is now a per-solve false-accept budget** (sequential
  correction over candidates actually tested), and `SolveResult.prob` is the
  corrected p-value. Weakly-evidenced solves that previously passed only on
  the old arithmetic's optimism may now fail — raise `match_threshold` (e.g.
  `1e-3`) to accept them explicitly. See *Changed — verification statistics
  recalibrated*.
- **Breaking (Rust):** `CentroidExtractionConfig.use_8_connectivity` is
  removed — detection is 8-connected by construction. Python is unaffected.
  See *Changed (breaking) — one run-length detection core*.
- **Serialized artifacts:** Python `SolveResult` pickles saved by 0.8.0 do
  not load — the `Solution` wire format changed (the write-only
  `image_width` / `image_height` fields were removed; the same values live in
  `Solution.camera_model`). Re-pickle after upgrading. Saved solver databases
  are unaffected. See *Changed (breaking)*.

### Changed (breaking) — one run-length detection core; `use_8_connectivity` removed

Both extraction paths now detect through a single run-length union-find core
(`sweep_runs`): a raster sweep turns the threshold predicate into horizontal
runs, merges 8-connected runs across rows, and hands each caller its regions
as run lists. The quality path no longer materializes a u8 mask or a u32
labels buffer (~10 MB at 2 Mpix) and its per-blob stages iterate run lists
in the same row-major order as before (moment sums bit-identical; TESS
calibration metrics unchanged to the last digit). Measured on 2048² TESS
frames: CCL extraction ~42 → ~26 ms (cumulative 71.5 → 26 ms this release),
fast path ~23 → ~15-20 ms.

**Breaking:** `CentroidExtractionConfig.use_8_connectivity` is removed —
run merging is 8-connected by construction (4-connectivity was never useful
for stars). The Python API is unaffected (it always used 8-connectivity).

### Changed — matched filter on by default, threshold auto-compensated

- **`matched_filter_sigma` defaults to `Some(1.5)`** (was `None`): every
  serious point-source detector convolves before thresholding; for a
  σ≈1.5 px PSF the peak-SNR gain is ~2× (≈0.75 mag more detection depth at
  the same false-positive rate), with a broad optimum (σ within ~2× of the
  true PSF width). Set `None` to threshold unfiltered.
- **The detection threshold is now scaled by the kernel's noise-suppression
  factor** (Σk² of the separable blur), so `sigma_threshold` means "sigmas
  of the noise actually present in the thresholded image" with the filter on
  or off — toggling the filter no longer requires retuning the threshold.
  `ExtractionResult.threshold` reports the threshold actually applied.
- **The filter now convolves the unclamped residual.** Previously it blurred
  the zero-clamped background-subtracted image, rectifying negative noise
  into a positive DC offset that silently loosened the effective threshold.

### Changed — sub-pixel peak refinement fits log intensity

The 3×3 parabola refinement (both extraction paths) now fits **log**
intensity when all nine background-subtracted samples are positive: a
Gaussian PSF is exactly quadratic in `ln(v)`, which removes the linear fit's
S-curve bias (~0.05–0.1 px at quarter-pixel peak phases). Blobs with
non-positive samples in the window keep the linear fit. Measured on the TESS
10-image multi-sector calibration: pooled fit residual 0.132 → 0.077 px
(−42%) and typical per-sector solve RMSE 2.5–2.9″ → 1.0–1.7″.

### Added — fast-path trail/streak rejection and covariance

`FastCentroidConfig` gains `max_pixels` (default 10000 — without it a
satellite trail or bloomed region becomes the *brightest* centroid handed to
the solver) and `max_elongation` (opt-in; moment-based elongation is noisy
for few-pixel regions). The fast path now accumulates intensity-weighted
second moments inline (merged through union-find), so it also populates
`Centroid.cov` like the CCL path. Note the coarse background grid already
absorbs structure larger than a block (~64 px) before these filters see it;
`max_pixels` matters for sharp bloomed regions, `max_elongation` for trails.

### Added — `border_margin`

Both extraction configs gain `border_margin` (default 0 = off): drop blobs
whose bounding box comes within the margin of an image edge. A star cut off
by the frame boundary has a truncated PSF, biasing its center-of-mass toward
the interior — a plausible but wrong position that previously only the 3×3
parabola refinement guarded against (by falling back to that biased CoM).

### Added — opt-in deblending, centroid-accuracy characterization

- **`deblend: DeblendMode`** (CCL path, default `Off`): a blended star pair
  produces one centroid at the flux-weighted midpoint — a wrong position the
  pattern hash will consume. `Reject` drops blobs with more than one distinct
  intensity peak (strict 8-neighborhood maxima above 30% of the blob peak,
  > 2 px apart); saturated blobs are exempt (plateau noise fakes maxima on a
  genuinely single star). Python: `deblend="off" | "reject"`.
- A deterministic ensemble test characterizes centroid accuracy across
  sub-pixel phases (bright stars: ~0.004–0.006 px RMSE at PSF σ 0.9–1.5 px)
  and guards against sub-pixel regressions.

### Added — extraction quality gates

- **`max_sharpness`** (both extraction configs): DAOFIND-style hot-pixel /
  cosmic-ray gate — rejects blobs whose peak sharpness
  `(peak − mean(8 neighbors))/peak` exceeds the limit, measured on the
  unfiltered background-subtracted image (so a matched-filter-smeared hot
  pixel is still caught). **Defaults to 0.9**, which passes any system whose
  PSF spans multiple pixels (a critically sampled PSF scores ~0.5, a strongly
  undersampled one ~0.85, a hot pixel ~1.0). Set `None` for severely
  undersampled data (PSF FWHM below ~1.5 px, e.g. resampled survey cutouts),
  where real stars are geometrically indistinguishable from hot pixels.
- **`saturation_level`** (both extraction configs): blobs whose peak reaches
  the sensor's saturation level skip quadratic sub-pixel peak refinement (a
  flat-topped or bloomed profile has no meaningful maximum) and keep the
  center-of-mass position. Off by default.

### Fixed — Python `calibrate_camera` accepts `SolveFailure` items

The Rust calibrate API accepts failed solves in its input slice and skips
them; the Python binding raised `TypeError` on any `SolveFailure` in the
list, forcing callers to filter (and mis-align their centroid lists). Mixed
lists now pass straight through. (Latent until this release — under the
recalibrated verification statistics, marginal solves can legitimately fail,
so mixed lists are the norm for the tiered calibration workflow.)

### Fixed — CCL-path noise estimator (`sigma_threshold` semantics)

`extract_centroids_from_raw` / `extract_centroids_from_image` (the CCL path)
underestimated the background noise by ~40%: sigma was computed as the RMS of
below-median pixels about their *own mean* — for Gaussian noise the lower
half-distribution has std ≈ 0.60σ — instead of about the *median*, whose
lower-half second moment equals the full variance (the estimator the
docstring described, and the one the fast path already used). In practice a
configured `sigma_threshold: 5.0` was really a ~3σ cut, the two extraction
paths applied materially different effective thresholds for the same setting,
and `ExtractionResult.background_sigma` was wrong as a diagnostic.

**Migration:** the CCL path's `sigma_threshold` now means true Gaussian
sigmas. To keep your previous effective detection depth, multiply configured
values by ≈0.6 (e.g. `5.0` → `3.0`); leave them unchanged to get the
threshold you had nominally been asking for. `FastCentroidConfig` is
unaffected (it was already correct).

### Changed — verification statistics recalibrated (solve-acceptance semantics)

The lost-in-space acceptance test was rebuilt around honest statistics; no
public API changed, but what the solver accepts (and how fast it fails) did:

- **Null model measured, not assumed.** The false-positive probability of a
  match count now uses the *measured* density of projected catalog stars over
  the frame with the correct π·r² disc area. The previous `num_nearby·mr²`
  under-predicted coincidence rates 2-3× per match — compounding to a latent
  false-accept vulnerability in dense fields (a wrong-attitude TESS candidate
  matching 113 stars by pure coincidence scored as a 10⁻¹⁴ certainty; it is
  now correctly rejected at p≈1).
- **Hypothesis stars aren't evidence.** The 4 pattern stars that formed the
  candidate are excluded from the binomial trials and successes (upstream
  tetra3's flat "−2" heuristic dropped). Tracking verification, whose hint is
  independent of the centroids, no longer takes any discount.
- **Sequential multiple-comparison correction.** Candidate `k` is accepted at
  `p·k < match_threshold` — a Bonferroni correction over candidates *actually
  tested* rather than the old division by the database pattern count, which
  over-corrected by 4-7 orders of magnitude and made clean sparse fields
  (< ~7 stars) mathematically unsolvable at any signal quality.
  `match_threshold` is now a per-solve false-accept budget (the total for a
  full search is bounded by a small logarithmic multiple);
  `Solution.prob` reports the corrected p-value.
- **Post-refinement re-verification.** Acceptance is decided by re-verifying
  the *refined* attitude at a radius tied to the refined RMSE. A true
  candidate's matches sit within a few RMSE (its p-value collapses by tens of
  orders when the radius tightens); a false candidate's coincidences are
  uniform across the search radius and cannot be aligned by the 3-DOF fit.
- **Robust to wrong FOV estimates at full speed.** Candidate vectors are
  rebuilt at the FOV measured from each matched pattern, so the FOV sweep
  collapses to a pattern-tolerance-derived step (usually a single value):
  a 15%-wrong FOV estimate now solves in ~36 µs instead of ~21-40 ms, and
  unsolvable fields fail in ~1.6 ms instead of ~12-29 ms (10° defaults).

Measured consequences (synthetic 10° harness, 1000 fields each): 6-star
fields 71% solvable and 8-star fields 92% (both 0% before by construction),
zero wrong-attitude accepts across 1500 pure-noise fields and all solved
scenarios. Real-sky solves that previously passed only by the old
arithmetic's optimism — e.g. heavily distorted TESS frames solved with an
uncalibrated camera model at tight `match_radius`, where the true-match rate
barely exceeds chance — may now fail or time out; raise `match_threshold`
(e.g. `1e-3`) to accept such weak evidence explicitly, or calibrate the
distortion first (the tiered calibration flow does this automatically).

### Changed (breaking)

- **`Solution.image_width` / `Solution.image_height` removed.** They were
  never read — `Solution.camera_model.image_width/height` carries the same
  values. This changes the postcard pickle wire format, so Python
  `SolveResult` pickles saved by 0.8.0 do not load; re-pickle after upgrading
  (see *Upgrading from 0.8*). Saved solver databases are unaffected.

- **`calibrate_camera` now returns `Result<CalibrateResult>`** (was
  `CalibrateResult`). It returns `Error::InvalidInput` instead of fabricating a
  camera model when the input has no successful solves, no parity consensus, or
  too few matched points for any fit to complete (previously these produced an
  identity model with a `0.1 rad` invented FOV or an `f64::MAX` RMSE, or
  panicked). Rust callers must handle the `Result`; the Python
  `SolverDatabase.calibrate_camera` now raises `ValueError` in these cases.
- **`PolynomialDistortion::new` takes 4 arguments** `(order, scale, a, b)`
  instead of 6 — the legacy inverse `ap`/`bp` coefficients are zero-filled
  (the model inverts numerically via Newton). The Python constructor keeps
  `ap_coeffs` / `bp_coeffs` as optional, ignored keyword arguments for
  backward compatibility. The struct fields persist for binary-format
  compatibility.
- **`solve_from_centroids` (Python)** no longer requires `fov_estimate_*` or
  `image_*` when a `camera_model` is given — the model already carries that
  geometry. They remain required when `camera_model` is omitted.
- Narrowed the public surface: `star_from_gaia` / `star_from_hipparcos`,
  `SolveConfig::pixel_scale`, and the `StarCatalog` spatial-index fields are
  now crate-private; `StarCatalog::{from_slice, query_stars_from_uvec}`,
  `query_stars` (now test-only), and `PolynomialDistortion::is_zero` were
  removed. `GaiaStar::{pmra, pmdec}` are now `f32` (were always-`Some`
  `Option`s). Unused `HipparcosStar` uncertainty/parallax/`v_i` fields dropped.

### Performance

- **CCL extraction ~40% faster** (71.5 → ~42 ms on 2048² TESS frames,
  matched filter on). Block medians are computed from a phase-staggered
  stride subsample; the materialized full-image background buffer and its
  separate subtraction passes are replaced by one fused pass that
  interpolates the block grid on the fly (writing the filter's unclamped
  input directly into the blur matrix); noise statistics run on subsampled
  bilinear residuals with the same estimator. The fast path hoists the
  row-constant half of its per-pixel bilinear out of the sweep (~10%).
- **Faster no-match / wrong-FOV solves.** The FOV sweep step doubled to
  `4·match_radius·fov` — measurements show verification tolerates the full
  match radius of midpoint scale error, so half the sweep values cover the
  same `fov_max_error` with identical solve rates. No-match fields drop
  28.6 → 12.8 ms and a 15%-wrong FOV estimate 40.8 → 21.1 ms on the profiling
  harness (10° FOV, ±2°).
- **Faster refinement (~14% off easy-solve latency).** Phase-D re-association
  projects catalog stars with a per-iteration rotation matrix instead of
  per-star TAN trig (mathematically identical), and prunes its cached cone
  list to stars that can still match while the cache is valid.
- Candidate-key enumeration skips key tuples that violate the catalog's
  ascending-ratio invariant (they can never hit); bounds the worst case for
  wide `match_max_error` settings.

### Fixed — 0.9 review pass

- **`matched_centroid_indices` now always index the caller's input slice.**
  When the solver dropped non-finite centroids it compacted the working list,
  so every reported index at or beyond a drop point was shifted — silently
  pairing wrong observed positions with catalog stars (and corrupting a
  `calibrate_camera` distortion fit built from them). Indices are now
  translated back through the drop map on both the LIS and tracking paths.
- **Lost-in-space now requires ≥ 5 centroids** (was ≥ 4). A 4-centroid field
  is all pattern stars with zero independent verification evidence, so it could
  never pass acceptance at any `match_threshold`; it now returns `TooFew`
  immediately instead of burning the whole FOV sweep to a silent `NoMatch`. The
  tracking (attitude-hint) path is unchanged.
- **`saturation_level` is compared against the raw sensor value on the default
  CCL path**, not the background-subtracted residual. A clipped star's residual
  peak sits below the clip level, so the saturation exemption never fired:
  `deblend = Reject` could split genuinely single bright stars on plateau
  noise, and the sub-pixel parabola ran on flat tops. Both extraction paths now
  behave identically for a given `saturation_level`.
- **The Gaia binary loader tolerates trailing bytes again.** The corrupt-header
  guard now requires *at least* `num_stars × 36` data bytes rather than exactly
  that — a `.bin` with padding/appended metadata (which 0.8 read fine) loads
  instead of failing with `InvalidCatalog`; the truncation guard is preserved.
- **`pattern_max_error` validation matches its documented `(0, 0.25]` range.**
  Values in `(0.25, 0.5]` previously passed and quantized every key dimension
  into a single degenerate bin; they are now rejected.
- **Python numpy centroid parsing is zero-copy for native-`f64` arrays** again
  (the common case): it tries a borrow first and only falls back to an
  `astype("float64")` copy for other dtypes.

### Fixed

- **Robustness sweep** — hardened panics, OOM, and silent-garbage paths
  reachable from ordinary input: centroid extraction rejects degenerate images
  / bad config instead of panicking; NaN-safe medians throughout; the solver
  caps the candidate-key enumeration (a large `match_max_error` could allocate
  gigabytes), bounds the hash-probe walk on corrupt databases, and drops
  non-finite centroids; `GenerateDatabaseConfig::validate` rejects
  index-corrupting parameters; the Gaia loader validates its header before
  allocating. Python bindings raise typed exceptions (`ValueError` / `IOError`
  / `TypeError`) instead of aborting, accept big-endian (FITS) and non-`f64`
  arrays, and normalize/validate `attitude_hint` quaternions and matrices.
- **Type stubs** reconciled with the bindings: removed the phantom
  `undistort_centroids` / `distort_centroids` functions and the nonexistent
  `SolveResult.distortion` property (added the real `camera_model`), fixed the
  `crpix` dtype and `@overload` forms, and added `SolveFailure` /
  `extract_centroids_fast` to `__all__`. `CatalogStar` now pickles.

## 0.8.0

> **Note on serialized artifacts.** 0.8.0 changes the on-the-wire format of
> several public types, so binary databases and pickles saved by 0.7.x do not
> load: `SolveResult` (now `Result<Solution, SolveFailure>`) and any
> `RadialDistortion` / `CameraModel` (new `RadialDistortion::center` field).
> Regenerate databases and re-pickle results after upgrading.

### Added

- **Fast single-pass centroid extractor — an "adequate star tracker" path.**
  `extract_centroids_fast` (Rust + Python) / `FastCentroidConfig` reads each
  pixel once: a cheap subsampled pre-pass builds a coarse background grid, then
  a single raster sweep thresholds against the interpolated background and
  groups lit pixels into connected regions via run-length + union-find,
  accumulating intensity-weighted moments inline and emitting one center-of-mass
  per region (with an optional 3×3 parabola peak refine). No convolution and no
  second pass, so it is memory-bandwidth-bound: **~4–5× faster** than the
  connected-component path single-threaded on 2048² TESS frames (~26–32 ms vs
  ~126 ms), with equal solve accuracy and ~0.1 px centroid agreement on bright
  stars. It trades faint-star sensitivity and tight sub-pixel accuracy for
  speed; `extract_centroids` / `extract_centroids_from_raw` stay the default and
  the right choice for calibration and faint-star work. Returns the same
  `ExtractionResult`, so it is a drop-in for `solve_from_centroids`.

### Changed

- **Centroid origin moved to the geometric image center `(W−1)/2`, `(H−1)/2`
  (was `W/2`, `H/2`).** Pixel centers sit at integer indices, so the geometric
  center — the intersection of the four central pixels for even dimensions, the
  middle pixel for odd — is at `(W−1)/2`, a half-pixel left of and above the old
  origin. This matches the FITS / astropy / astrometry.net / OpenCV convention,
  removing a ~½-pixel (~130–250″ on a wide-field camera) bias when comparing
  tetra3rs solutions against those tools or feeding them a `crpix` derived from
  one. The old origin was internally consistent, so tetra3rs-only solves were
  not biased; returned centroid coordinates and the boresight now shift by half
  a pixel toward the geometric center. Applies to `extract_centroids`,
  `extract_centroids_fast`, and any caller-supplied centroids (which should now
  use the `(W−1)/2` origin). The docs' coordinate page now also states
  explicitly that the solved attitude is the direction at the **center pixel**,
  not the optical/distortion axis. (Issue #28.)

- **Radial calibration rewritten as a standard camera-intrinsics fit
  (OpenCV-style).** `calibrate_camera(model = Radial)` now jointly fits the
  optical-axis position `(cx, cy)`, a focal-scale factor `γ`, and the
  Brown-Conrady coefficients `(k1, k2, k3, p1, p2)` — the same parameter
  set as OpenCV's `calibrateCamera` (free principal point, free focal
  length). Two structural problems in the old fit are gone:
  - *No scale degree of freedom.* The fit was anchored to the focal length
    implied by the median solve FOV — a whole-field average biased by the
    very distortion being fit (≈0.8% on TESS, ~12 px at the field corner).
    Brown-Conrady has no linear term, so the bias had to be mimicked by the
    cubic+ terms, making results hypersensitive to the FOV estimate. The
    fitted `γ` now absorbs it exactly and is folded into
    `CameraModel::focal_length_px` (with the exact coefficient rescaling
    `kᵢ → kᵢ/γ^2i`, `p → p/γ`).
  - *Optical center pinned to the image center.* The old `(cx, cy)`
    regularizer pulled the distortion center to `[0, 0]`, which is wrong on
    mosaic cameras (TESS, Kepler, Rubin, …) where the optical axis lies far
    off any single detector — near a CCD corner for TESS Camera 1 CCD 1.
    The center is now free, with only a tiny tie-breaking prior for the
    weak-distortion case where it is genuinely unidentifiable.

  The fitter also normalizes coordinates internally (the Jacobian otherwise
  spans ~20 orders of magnitude and LM fails at real-camera scales) and no
  longer discards the fit when the LM iteration budget is reached. On the
  TESS 10-image radial calibration this takes the pooled fit from 6.3 px
  (or 3.0 px with 60% of points clipped, depending on the FOV estimate) to
  1.8 px with 80% inliers, and post-calibration re-solves agree with the
  FITS WCS to better than 90″ on all sectors.

- **`RadialDistortion` carries its own distortion center.** New
  `center: [f64; 2]` field (pixels, image-center-origin frame; default
  `[0, 0]`) and `with_center()` constructor. The distortion center (optical
  axis) is now distinct from `CameraModel::crpix` (the tangent-plane
  projection origin): radial calibrations leave `crpix = [0, 0]` and store
  the fitted optical-axis position in the model, so solve-time geometry
  (FOV-radius catalog queries, pattern scales) is unaffected even when the
  optical axis sits 1400 px off the image center. **Breaking:** saved
  binary/pickled `RadialDistortion` (and containing `CameraModel`) blobs
  from earlier versions do not load. Python: `RadialDistortion` gains a
  `center=(cx, cy)` constructor argument and property.

- **Calibrated `focal_length_px` is now tan-consistent.** Both calibrate
  paths previously recorded `focal_length_px = image_width / fov_rad`
  (small-angle) while fitting distortion against tan-projected ideal points
  (`f = (w/2)/tan(fov/2)`), a ~0.3% inconsistency at TESS's 11.7° FOV. Both
  now use the tan form, matching `CameraModel::from_fov` / `fov_rad()`.

- **Multi-image calibration excludes parity-outlier solves.** Parity is a
  physical property of the camera, so all images in a calibration set must
  agree; a lone opposite-parity solve is a false (mirror-image) match whose
  star correspondences would poison the pooled distortion fit. Images
  disagreeing with the majority parity are now excluded (debug-logged), and
  the median-FOV/parity consensus is computed over agreeing solves only.
  (Seen on TESS sector 14, which falsely solves as a mirror image with 427
  "matches" at rmse 275″.)

### Fixed

- **Parity-flipped solves no longer return a corrupt attitude.** For images
  requiring a parity flip (`parity_flip = true`), the final rotation was
  rebuilt from `(θ, CRVAL)` with a parity branch that produced a *reflection*
  (det −1) instead of a rotation. `Quaternion::from_rotation_matrix` cannot
  represent a reflection, so `qicrs2cam` was meaningless (a non-unit,
  near-identity quaternion), and the recomputed residual statistics
  (`rmse_rad`, `p90e_rad`, `max_err_rad`) were degrees-level garbage. The
  exported `cd_matrix` encoded the roll with the wrong sign convention
  (`ps·diag(−1,1)·R(θ)` instead of `ps·R(θ)·diag(−1,1)`). The bug was
  invisible to the test suite because no test exercised an end-to-end
  parity-flipped solve (the skyview tests pre-correct parity and the
  synthetic fields are proper). Now: the rotation/quaternion always describe
  the x-negated (proper) working frame — `Solution.parity_flip` records the
  mirror, exactly as documented — `cd_from_theta` emits the matching CD
  convention, and `wcs_to_rotation` returns the proper working-frame rotation
  for `det(CD) < 0` inputs. `Solution::pixel_to_world` / `world_to_pixel`
  were already consistent and are unchanged. Covered by new unit tests and an
  end-to-end mirrored-field integration test (`test_parity_flipped_solve`).
  Rust API note: `rotation_from_theta_crval` no longer takes a `parity_flip`
  argument (the rotation does not depend on it).

### Breaking changes

- **Python: `solve_from_centroids` returns a `SolveFailure` object instead of
  `None` on failure.** The new object is *falsy* — `if result:` keeps working
  — and carries `status` (`'no_match'` / `'timeout'` / `'too_few'`) and
  `solve_time_ms`, which were previously discarded. Code using
  `result is None` / `result is not None` must switch to truthiness checks
  (`if not result:` / `if result:`). `SolveFailure` pickles like the other
  public types.

- **`SolveResult` is now `Result<Solution, SolveFailure>` (Rust).** The old
  struct-of-`Option`s is replaced by a `Solution` whose fields are all
  guaranteed present (`qicrs2cam: Quaternion`, `fov_rad: f32`, `camera_model:
  CameraModel`, `cd_matrix`, `crval_rad`, `theta_rad`, …) and a `SolveFailure
  { status, solve_time_ms }` for the `NoMatch` / `Timeout` / `TooFew`
  outcomes (`SolveStatus::MatchFound` is gone — success is the `Ok` arm).
  `Solution::pixel_to_world` is now infallible and the legacy CD-matrix and
  quaternion+FOV fallback transform paths are deleted (every `Solution`
  carries a camera model and θ). `calibrate_camera` and the distortion
  fitters still accept failed solves in their input slices and skip them.
  **Python:** `solve_from_centroids` still returns a `SolveResult` object or
  `None`; the change is that its attributes (`fov_deg`, `num_matches`,
  `rmse_arcsec`, `camera_model`, `cd_matrix`, …) are no longer `Optional`,
  and scalar `pixel_to_world` always returns a tuple. Pickles of
  `SolveResult` objects from earlier versions do not load (wire format
  changed).

- **`SolveConfig`: `CameraModel` is now the single source of camera geometry
  (Rust).** The redundant `fov_estimate_rad`, `image_width`, and
  `image_height` fields are removed; the FOV estimate, image dimensions, and
  pixel scale all derive from `camera_model` (new accessors
  `fov_estimate_rad()`, `image_width()`, `image_height()`, and a
  `SolveConfig::with_camera_model()` constructor). `SolveConfig::new(fov, w,
  h)` is unchanged and remains the easy path. This removes the possibility of
  an inconsistent config (e.g. a calibrated model alongside a mismatched FOV
  estimate — previously the estimate silently won) and deletes the tracking
  path's "is the camera model real?" heuristic. The **Python API is
  unchanged** except for precedence: when `camera_model=` is passed, its focal
  length and dimensions are now authoritative and `fov_estimate_*` /
  `image_shape` are ignored (previously `fov_estimate` set the pixel scale
  even alongside a model).
- **Removed `SolveConfig::refine_iterations` (Rust) and the
  `refine_iterations=` kwarg of `solve_from_centroids` (Python).** The field
  was never read by the solver — the documented "number of iterative SVD
  refinement passes" did not exist; refinement depth has always been governed
  by the WCS refinement loop's internal convergence (stable match set, capped
  at 10 outer iterations). Setting it had no effect, so removal does not
  change solver behavior. Python callers passing `refine_iterations=` must
  drop the argument.

### Solver robustness

- **No more panics on degenerate input.** A failed SVD during attitude
  estimation (e.g. pathological/duplicate centroids) now skips the candidate
  (lost-in-space) or fails the hinted solve (tracking) instead of panicking;
  NaN residuals no longer panic the WCS refinement's median/MAD statistics;
  an empty pattern catalog (corrupt database) returns `NoMatch` instead of
  dividing by zero.
- **FOV sweep no longer aborts early on cluster-buster `TooFew`.** The
  post-thinning "too few pattern centroids" condition depends on the FOV being
  tried; the sweep now continues to other FOV values instead of returning.
  A genuine input of fewer than 4 centroids still returns `TooFew` immediately.
- **Verification cone sized for portrait images.** The catalog query radius
  now uses the true image diagonal (`sqrt(1 + (h/w)²)`, floored at the
  historical 1.42 factor) instead of assuming a square/landscape sensor.
- **Tracking solves are now aberration-consistent.** With
  `observer_velocity_km_s` set, the hinted-solve path previously matched
  against aberration-corrected star positions but ran the WCS refinement and
  residual statistics against the raw catalog vectors. Tracking now uses the
  same corrected unit-vector slice as lost-in-space end to end.

### Internal

- Deduplicated solver helpers: one `separation_for_density` (was copied in
  `solve.rs` and `database.rs`), one generic pattern centroid-distance sort,
  shared `focal_length_from_fov` / `pixel_scale_from_fov` for the pinhole
  formula, and extracted `compute_residuals` / `ls_fit_once` in the WCS
  refinement (the residual loop appeared three times, the LS pass twice).
  Brightness sorting is hoisted out of the per-FOV loop; the tracking solver's
  catalog-index reverse lookup is gone (match pairs now carry local indices).
- One verification and one refinement pipeline: the catalog-projection /
  greedy-match / binomial-FPR verification block and the WCS-refine +
  finalize tail are now single shared methods (`verify_attitude`,
  `refine_and_finalize`) used by both the lost-in-space and tracking paths
  (each previously had its own copy). The final attitude is derived directly
  from the constrained-fit parameters (θ, CRVAL, parity) and the locked pixel
  scale instead of synthesizing a CD matrix and decomposing it back into a
  rotation and FOV.

## 0.7.4

### New features

- **`SolverDatabase.print_parameters()` / `.parameters` (Python).** Report the
  settings a database was generated with — stars per FOV
  (`verification_stars_per_fov`), lattice field oversampling, patterns per
  lattice field, pattern quantization (`pattern_bins` / `pattern_max_error`),
  FOV range, magnitude limit, catalog/proper-motion epochs, and star/pattern
  counts. `print_parameters()` prints a grouped, human-readable dump;
  `parameters` returns the same data as a dict. Both read the stored database
  properties, so they reflect the actual `.bin` on disk.

### Internal

- Silenced pre-existing clippy `too_many_arguments` warnings on the
  kwargs-heavy PyO3 functions and a redundant `Centroid` conversion in the
  Python crate.

> **Note on 0.7.3 / PyPI.** The crates.io `tetra3` 0.7.3 release is valid. The
> `tetra3rs` **PyPI** 0.7.3 wheels were withdrawn after a partial upload, and
> PyPI permanently reserves used filenames — so the Python package skips from
> 0.7.2 to 0.7.4. Everything in 0.7.3 below ships in the 0.7.4 wheels.

## 0.7.3

### New features

- **Optional `parallel` cargo feature: multi-threaded centroid extraction.**
  Rayon-based parallelism for the extraction hot paths, off by default.
  Parallelizes the dominant local-background stage (independent block medians +
  bilinear interpolation rows, ~60% of extraction wall-clock) and the full-image
  element-wise background-subtraction maps. Also enables numeris's rayon
  `imageproc` paths, so the optional matched-filter Gaussian blur runs threaded.
  Results are bit-identical to the sequential build. Measured ~1.9× (sparse
  2-Mpix field) / ~1.45× (dense TESS field, ~37k blobs) on 8 cores. Build with
  `--features image,parallel`. Connected-component labeling (sequential in
  numeris) and the small per-blob centroid loop are left single-threaded.

### Performance

- **Faster solves, behavior-identical** (no algorithm or output changes):
  - In the lost-in-space path, derive the parity-flipped rotation algebraically
    (negate the first row of `R = U·Vᵀ`) instead of running a second SVD when the
    initial rotation has `det < 0`. The old "still a reflection → skip" branch is
    provably dead and removed.
  - In `wcs_refine`, hoist heap allocations out of hot paths: the greedy pixel
    matcher reuses a `MatchScratch` buffer set across calls, `residual_median_sigma`
    uses `select_nth_unstable_by` (partial selection) instead of a full sort, and
    the Phase-D `predicted` list and match scratch are allocated once before the
    outer refinement loop.

### Internal

- Bumped `numeris` 0.5.11 → 0.5.12 (provides the rayon `imageproc` paths).

## 0.7.2

Performance and tooling. No breaking public API changes.

### New features

- **Faint-magnitude Gaia downloads via Flatiron flathub (issue #30).**
  New script `scripts/download_gaia_flatiron.py` queries the Flatiron
  Institute's [flathub](https://flathub.flatironinstitute.org/) service
  to pull Gaia DR3 past G ≈ 11.5. ESA TAP's anonymous async output is
  capped at a hard 3,000,000 rows per job (server-side, advertised in
  `/capabilities`), so the existing `scripts/download_gaia_catalog.py`
  effectively tops out at G ≈ 11.5 (G < 12 is 3.09M rows, just over the
  cap). flathub has no such cap and serves the full Gaia DR3 catalog.
  Both scripts share the Hipparcos-2 bright-star merge and produce
  byte-compatible `.bin` / `.csv` output, so the new downloader is a
  drop-in when fainter limits are needed. flathub is not on PyPI —
  install from the [upstream repo](https://github.com/flatironinstitute/flathub/tree/prod/py).

### Performance

- **~1.7× faster typical lost-in-space solves** (10° clean field) versus 0.7.1,
  with identical accuracy. The gain is concentrated in `wcs_refine`, which
  dominates a successful solve:
  - Precompute each catalog star's `(ra, sin dec, cos dec)` once and hoist the
    CRVAL `sin`/`cos` out of the per-star projection loop.
  - Cache the Phase-D re-association catalog cone query and its projected star
    set, reusing it across refinement iterations instead of re-querying every
    pass (catalog query drops from ~4× to ~1× per solve).
  - Detect a stable match set and stop one iteration earlier.
  - Prune off-frame predicted stars and use a bitset (not a hash set) in the
    pixel matcher.

### Internal

- Solver core simplified (dead-code removal, deduplication) with no change to
  public behavior.
- New optional `profile` cargo feature adds zero-cost (when disabled)
  thread-local leaf timers to the solve path, plus an `examples/profile_solve`
  harness. See `CLAUDE.md`.
- Added Python test coverage for the tangential `RadialDistortion` parameters
  (`p1`, `p2`).

### Other

- New [Star Catalog](https://tetra3rs.dev/getting-started/catalog/)
  documentation page covering the pre-built download, both `scripts/`
  downloaders (including the ESA TAP 3M-row cap and account-registration
  workaround), the Hipparcos bright-star merge, and the on-disk format.
  Signposted from the README, the installation page, and the `tetra3`
  crate top-level doc.

## 0.7.1

### New features

- **Optional Gaussian matched filter in centroid extraction.** New
  `CentroidExtractionConfig::matched_filter_sigma: Option<f32>` field
  (Python keyword `matched_filter_sigma`, default `None`). When set, the
  background-subtracted residual is convolved with a separable Gaussian
  (kernel truncated at 3σ, replicate border) before thresholding. The
  filtered image is used **only** to form the detection mask — centroid
  positions and intensities are still measured on the unfiltered residual,
  so photometry is unaffected. Boosts point-source SNR for detection in
  noisy or dense fields. Consider lowering `sigma_threshold` to 2.5–3.0
  when enabled.

## 0.7.0

### Breaking changes

- **Serialization format changed from rkyv to [postcard](https://docs.rs/postcard).**
  Existing `.rkyv` databases saved by 0.6.x or earlier will fail to load.
  Regenerate via `SolverDatabase::generate_from_gaia(...)` (Rust) or
  `generate_from_gaia(...)` (Python). The same applies to any `.rkyv`
  `CameraModel` files saved with `CameraModel::save_to_file`. First-use
  regeneration is automatic for Python users whose databases live in the
  `gaia-catalog` package cache.
- **File-extension convention changed from `.rkyv` to `.bin`** in docs and
  examples. Existing user files keep working under any extension; the
  rename is cosmetic.
- **Pickle format changed.** Pickled `tetra3rs` objects (`SolverDatabase`,
  `CameraModel`, `SolveResult`, `CalibrateResult`, `ExtractionResult`,
  `Centroid`, `RadialDistortion`, `PolynomialDistortion`) now round-trip
  through postcard. Pickles produced by 0.6.x will not unpickle.
- **`SolverDatabase::pattern_catalog` is now a flat `Vec<PatternEntry>`-backed
  `PatternCatalog`** — the 0.6.0 sharding workaround for rkyv's 2 GB
  relative-offset limit is gone (postcard has no such limit). Public
  access via `.get(idx)` / `.get_mut(idx)` / `.len()` / `.is_empty()` is
  unchanged; the `PatternCatalog::SHARD_SIZE` constant and the
  `shards: Vec<Vec<PatternEntry>>` field are removed.
- **`SolverDatabase::to_rkyv_bytes` renamed to `SolverDatabase::to_bytes`.**

### New features

- **Lighter dependency footprint.** Around 10 crates removed from the build
  (`rkyv`, `rkyv_derive`, `bytecheck`, `bytecheck_derive`, `munge`,
  `munge_macro`, `ptr_meta`, `ptr_meta_derive`, `rancor`, `rend`,
  `simdutf8`) in exchange for 3 (`postcard`, `serde`, `serde_derive`).
- **Database files are portable.** postcard has a published wire-format
  spec, so a database written on one platform can be read by any
  postcard implementation — no longer Rust-locked.

### Other changes

- **`numeris` bumped to 0.5.11** for native serde support
  (`Matrix<T, M, N>`, `Quaternion<T>`), `imageproc` connected-component
  labelling, and `optim::least_squares_lm_dyn` (Levenberg-Marquardt on
  dynamic-size residual vectors — used by the new Brown-Conrady fit).
  The 0.4 → 0.5 bump was a minor API change at call sites:
  `Matrix::vecmul(&v)` is now the `*` operator (`m * v`), and
  `DynMatrix::zeros(rows, cols, fill)` / `DynVector::zeros(n, fill)`
  lost their fill argument (`zeros(rows, cols)` / `zeros(n)`). All
  internal call sites are updated.
- **`csv` dependency dropped.** The Gaia DR3 CSV reader
  (`catalogs::gaia::read_gaia_csv`) is now a hand-rolled ~30-line parser
  for the fixed 9-column unquoted schema produced by
  `scripts/download_gaia_catalog.py`. The CSV path through
  `SolverDatabase::generate_from_gaia` continues to work as before, just
  without the `csv`/`csv-core`/`bstr`/`aho-corasick`/`regex-automata`
  transitive tree.
- **Latent bug fix in CSV catalog loading.** The previous `csv::Reader`
  call had a stray `.skip(1)` after the header was already auto-skipped,
  silently dropping the brightest star from any CSV-loaded Gaia catalog.
  The new parser only skips the header line, so CSV-loaded catalogs now
  correctly include all stars. Binary (`.bin`) catalogs were never
  affected.
- **Public Rust function `extract_centroids(path, &config)` removed.**
  This was a 5-line convenience wrapper around `image::open(path)?` +
  `extract_centroids_from_image(...)`. Removing it lets the `image` dep
  drop its `jpeg` / `png` / `tiff` format features, killing a sizeable
  decoder dep tree (`zune-jpeg`, `png`, `tiff`, `fdeflate`, `weezl`,
  `moxcms`, `byteorder-lite`, `half`, `color_quant`, …). Callers who
  want file-path convenience should decode the file themselves with
  whichever `image` feature flags they need, then call
  [`extract_centroids_from_image`]. The Python
  `tetra3rs.extract_centroids(numpy_array, ...)` API is unchanged — it
  goes through `extract_centroids_from_raw` and never used the file path.
- **Connected-component labelling delegated to
  [`numeris::imageproc`](https://docs.rs/numeris).** The hand-rolled
  two-pass union-find in `centroid_extraction.rs` (~110 lines) is
  replaced by `numeris::imageproc::connected_components_with_label_buffer`,
  which also supplies per-blob area and bounding box. The
  `imageproc` feature on numeris is activated only when tetra3's
  `image` feature is on (`image = ["dep:image", "numeris/imageproc"]`),
  so non-image users don't pay for it. Intensity-weighted centroiding
  with per-blob local-background annulus refinement remains in tetra3.
- **`tracing-subscriber` moved from `[dependencies]` to
  `[dev-dependencies]`.** Library code emits log events via the
  lightweight `tracing` macros; configuring a subscriber is the
  application's job, not the library's. Downstream crates depending on
  `tetra3` no longer compile `tracing-subscriber` and its ~10
  transitive crates (`matchers`, `regex-automata`, `regex-syntax`,
  `nu-ansi-term`, `sharded-slab`, `lazy_static`, `smallvec`,
  `thread_local`, `tracing-log`).
- **`anyhow` replaced with a typed `tetra3::Error` enum.** All public
  functions that previously returned `anyhow::Result<T>` now return
  `tetra3::Result<T>` (alias for `Result<T, tetra3::Error>`). The
  `Error` enum has four variants — `Io`, `Postcard`, `InvalidCatalog`,
  `InvalidInput` — letting callers `match` on specific failure modes
  instead of receiving an opaque trait object. Implements `Display`,
  `std::error::Error`, and `From<std::io::Error>` /
  `From<postcard::Error>`. Built with `thiserror` (already in the dep
  tree via `postcard → cobs`, so no net dep added). The `anyhow` crate
  is dropped.
- **`SolverDatabase::generate_from_star_list` is now infallible** —
  signature changed from `anyhow::Result<Self>` to `Self`. The function
  never had any error sources; the `Result` wrapper was vestigial.
- **CSV catalog input dropped from `generate_from_gaia`.** The function
  previously branched on file extension / magic bytes between `.csv`
  and `.bin` paths; the `.csv` branch is gone. Callers must supply a
  `.bin` Gaia binary catalog (the format the bundled `gaia-catalog`
  package ships). The `read_gaia_csv` helper and its in-tree CSV
  parser are removed.
- **Stale Python `.pyi` stubs cleaned up.** The type stubs previously
  advertised `SolverDatabase.fit_radial_distortion()` /
  `.fit_polynomial_distortion()` methods that had no matching PyO3
  bindings (the stubs had drifted). Removed. The Rust functions
  `tetra3::distortion::fit::fit_radial_distortion` and
  `tetra3::distortion::fit::fit_polynomial_distortion` remain
  available as standalone Rust API.
- **`calibrate_camera` is now model-agnostic.** It can fit either a SIP
  polynomial (the existing default) or a full Brown-Conrady distortion
  model (radial `k1, k2, k3` + tangential `p1, p2`, plus an
  optical-axis offset `(cx, cy)` jointly fit). Selected via
  `CalibrateConfig::model: DistortionModelType { Polynomial { order } | Radial }`.
  Both models go through the same alternating per-image
  WCS-refinement + global-fit pipeline. Python: pass `model="radial"`
  (or `"polynomial"`, the default) to
  `SolverDatabase.calibrate_camera()`.
  Breaking: the Rust `CalibrateConfig::polynomial_order: u32` field is
  replaced by `model: DistortionModelType`; callers need to update
  construction to
  `CalibrateConfig { model: DistortionModelType::Polynomial { order: 4 }, .. }`.
- **`RadialDistortion` extended to full Brown-Conrady** — the struct
  now carries `p1, p2` tangential / decentering coefficients in
  addition to `k1, k2, k3`. Setting `p1 = p2 = 0` reduces to pure
  radial Brown-Conrady (the historical default). New constructor
  `RadialDistortion::with_tangential(k1, k2, k3, p1, p2)`;
  `RadialDistortion::new(k1, k2, k3)` keeps the same signature and
  sets `p1 = p2 = 0`. `distort` and `undistort` use the full forward
  model; the inverse uses 2D Newton iteration on the forward
  Jacobian. Python `tetra3rs.RadialDistortion(...)` accepts `p1, p2`
  kwargs and exposes them as properties.
  Breaking: pickle format changes (struct gained two fields). Rebuild
  any persisted radial distortion models.
- **Centered radial fit (`fit_radial_centered_sigma_clip`) uses
  numeris's nonlinear LS.** The 7-parameter joint fit
  `(cx, cy, k1, k2, k3, p1, p2)` dispatches to
  `numeris::optim::least_squares_lm_dyn` (Levenberg-Marquardt on
  dynamic-size residuals). Replaces ~250 lines of hand-rolled LM
  (normal equations + accept/reject loop + line search) with closures
  for residual and Jacobian. Mild regularization on `(cx, cy)` is
  implemented by augmenting the residual vector with two extra rows
  (`√μ·cx`, `√μ·cy`) — same cost shape as before, no special LM
  features needed. The polynomial fit is already a *linear* LS via
  `DynMatrix::solve_qr` and stays unchanged.
- **Direct `num-traits` dep removed.** tetra3's own code never used
  `num_traits`; it was pulled in transitively via `numeris` regardless.
  Manifest cleanup only.

### Notes on performance

postcard is not zero-copy, so loading a database now does an actual
deserialization pass instead of validating an archived layout in place.
In practice this is a non-regression: prior versions also fully
deserialized via `rkyv::from_bytes` rather than using zero-copy
`rkyv::access`, so nothing in the existing code path was benefiting from
zero-copy. If you need true mmap-style loading for very large databases
in the future, that's a separate change to consider.

### Internal

- `src/rkyv_numeris.rs` removed entirely. With numeris 0.5.10's native
  serde implementations of `Matrix<T, M, N>` and `Quaternion<T>`,
  `Option<Matrix2<f32>>` and `Option<Quaternion<f32>>` fields derive
  `Serialize` / `Deserialize` directly — no adapter shims needed.

## 0.6.1

### Fixes

- **`import tetra3rs` no longer crashes when installed without package
  metadata (issue #22).** The module-level call to
  `importlib.metadata.version("tetra3rs")` raised `PackageNotFoundError`
  in environments where the package is on `sys.path` without a
  corresponding `.dist-info/` directory — e.g. Pi OS image builds that
  copy the source tree instead of running `pip install`. `__version__`
  now falls back to `"0.0.0+unknown"` when metadata is unavailable.
  Properly pip-installed users see the correct version string as before.

## 0.6.0

### Fixes

- **Multiscale databases no longer crash on `save_to_file` (issue #13).** Databases
  covering a wide range of field-of-view scales (e.g. 0.5°–5°) can generate hash
  tables larger than 2 GB, which overflowed rkyv's default 32-bit relative-offset
  limit and caused serialization to panic with *"out of range integral type
  conversion attempted"*. The pattern catalog is now stored as
  [`PatternCatalog`](https://docs.rs/tetra3/0.6.0/tetra3/solver/struct.PatternCatalog.html)
  — a sharded container that splits its backing storage into independently-archived
  chunks of up to ~770 MB each. Probe logic is unchanged in spirit (one additional
  L1-resident dereference per probe, effectively zero runtime cost).

### Breaking changes

- **`.rkyv` database file format bumped.** Existing cached `.rkyv` files saved
  with 0.5.x or earlier will fail to load under 0.6.0. Regenerate via
  `SolverDatabase::generate_from_gaia(...)` (Rust) or
  `generate_from_gaia(...)` (Python). First-use regeneration is automatic for
  Python users whose databases live in the `gaia-catalog` package cache.
- **`SolverDatabase::pattern_catalog` field type** changed from
  `Vec<PatternEntry>` to `PatternCatalog`. Access slots via `.get(idx)` /
  `.get_mut(idx)` rather than `[idx]`; the hash-probe loop in user code (if
  any — most users don't access this field directly) needs a one-line update.
- **Python: upper-bounded `gaia-catalog<1.0`.** The bundled Gaia binary
  catalog format is unchanged in 0.6.0 and still works with
  `gaia-catalog` 0.1.x. Adding this upper bound is a forward guard: if
  `gaia-catalog` ever ships a breaking binary-format change under a
  `1.0` release, this prevents it from silently being installed under a
  `tetra3rs 0.6.x` that wouldn't be able to read it. No-op for users
  today. Older `tetra3rs` releases don't pin `gaia-catalog` at all —
  we'd protect 0.5.x users similarly via a future `0.5.2` patch if a
  breaking `gaia-catalog` release ever lands.

## 0.5.1

### New features

- **Tracking-mode solving via attitude hint.** `SolveConfig` gains an optional `attitude_hint: Option<Quaternion>` (plus `hint_uncertainty_rad` and `strict_hint`). When set, the solver projects catalog stars near the hinted boresight, nearest-neighbor matches them to centroids, runs Wahba SVD, and reuses the existing verification + WCS refine path — skipping the 4-star pattern hash entirely. Succeeds with as few as 3 matched stars (lost-in-space needs 4), is robust to pattern-hash failures from sparse / low-SNR fields, and on failure falls back to lost-in-space unless `strict_hint` is set. Intended for video-rate star trackers chaining solves frame-to-frame.

### Fixes

- `SolveResult::camera_model` now has `image_width` / `image_height` populated from the input `SolveConfig`. Previously these were inherited from `config.camera_model`, so a solve config built with `..Default::default()` (without explicitly constructing a `CameraModel`) would leave them zero, breaking downstream code that consumes the result's camera model.

### Other

- `CLAUDE.md` added to the repo root (guidance for Claude Code sessions in this repo).

## 0.5.0

### Precision improvements

- **True-pinhole pixel scale throughout.** The solver previously used the small-angle approximation `pixel_scale = fov / image_width` internally while storing `focal_length_px = (W/2) / tan(fov/2)` (true pinhole) on the result. At finite FOV the two differ by ~0.5%, producing ~100″ residuals at field corners if downstream code mixed them. The internal pipeline (`solve.rs`, `wcs_refine.rs`, `SolveConfig::pixel_scale`, distortion calibration, synthetic test generators) now uses `1/f` everywhere.
- **Newton iteration for polynomial undistort.** `PolynomialDistortion::undistort` now solves the forward polynomial by Newton iteration (2-4 iterations to machine precision) instead of evaluating a separately-fit inverse polynomial. A finite-order inverse polynomial cannot perfectly invert a finite-order forward polynomial, and the resulting asymmetry error amplified at field corners under tight match radii. Newton is exact (limited only by forward polynomial expressiveness) and eliminates the asymmetry.
- **TESS multi-image calibration**: average agreement with FITS WCS dropped from 0.81″ to **0.42″** across 10 sectors, with every sector improved or equal. Sector 17 specifically: RMSE 5.25″ → 2.56″.

### Breaking changes

- **`wcs_to_rotation` return value** — the returned FOV is now the *angular* FOV `2·atan(ps·W/2)` rather than the linear `ps·W`. Matches the convention of `fov_estimate_rad` elsewhere. Affects any external code calling this function directly; internal callers are all updated.
- **Removed `term_pairs_range` / `num_coeffs_range`** from `distortion::polynomial`. These `pub` helpers were unused in-tree and had no external users we're aware of.
- **`PolynomialDistortion::{ap_coeffs, bp_coeffs}`** are retained in the struct for binary-format compatibility but are zero-valued in any model produced by this crate. `fit_inverse_poly_ls` removed.

### Other

- `SolveConfig::pixel_scale()` return value is now `1/f` (true pinhole) instead of `fov / W` (linear); the two differ by ~0.5% at 15° FOV.

## 0.4.1

### New features

- **CameraModel save/load.** Added `save_to_file()` and `load_from_file()` methods to `CameraModel` for persisting camera intrinsics (including distortion) to disk using rkyv serialization. Available in both Rust and Python — models saved from one language can be loaded in the other.

### Other

- Added Gaia DR3 and Hipparcos 2 catalog attribution to LICENSE.

## 0.4.0

### Breaking changes

- **Gaia DR3 is now the default (and always-included) star catalog.** The `gaia` feature flag has been removed; Gaia support is always compiled in. Hipparcos support is now behind an optional `hipparcos` feature flag.
- **`Star.id` changed from `u64` to `i64`** to support negative source IDs for Hipparcos gap-fill stars in the merged catalog. This affects `matched_catalog_ids` arrays (`np.int64` in Python) and `get_star_by_id()`.
- **`generate_from_hipparcos` removed from Python bindings.** Use `generate_from_gaia()` instead.
- **Python dependency changed** from `hipparcos-catalog` to `gaia-catalog`.

### New features

- **Gaia DR3 + Hipparcos merged catalog.** The merged catalog uses Gaia for most stars and fills in the brightest stars (G < 4) from Hipparcos where Gaia saturates. Hipparcos positions are propagated from J1991.25 to the Gaia epoch (J2016.0).
- **Compact binary catalog format (.bin).** A custom 36-byte-per-star binary format (header + packed structs) reduces catalog size from 77 MB (CSV) to 17 MB. `generate_from_gaia()` auto-detects CSV vs binary format.
- **`gaia-catalog` PyPI package.** The merged catalog is bundled as a lightweight Python package (~15 MB wheel). `generate_from_gaia()` with no arguments automatically uses the bundled catalog.
- **`generate_from_gaia()` accepts optional `catalog_path`.** When `None` (default), uses the bundled `gaia-catalog` package. Accepts both `.csv` and `.bin` files.
- **`scripts/download_gaia_catalog.py`** downloads Gaia DR3 via TAP, merges with Hipparcos 2, and outputs either CSV or binary format (determined by file extension).

### Improvements

- **Switched from `nalgebra` to `numeris`** for linear algebra. `numeris` is a lightweight, `no_std`-compatible pure-Rust library for matrix/vector/quaternion operations, reducing dependency weight and improving suitability for embedded targets.
- **TESS test reliability.** Changed `match_radius` from `0.01` to `0.005` across TESS tests, fixing a false-match failure on the sparse field image.
- Removed `zip` dev-dependency (tests now download `.bin` directly instead of extracting from `.zip`).

### Migration guide

**Rust:**

Download the pre-built merged catalog (~17 MB, 482k stars to G-band magnitude 10, Gaia DR3 + Hipparcos bright-star gap-fill):
```sh
curl -o data/gaia_merged.bin "https://storage.googleapis.com/tetra3rs-testvecs/gaia_merged.bin"
```

Or generate your own with a custom magnitude limit:
```sh
python scripts/download_gaia_catalog.py --mag-limit 12.0 --output data/gaia_merged.bin
```

```rust
// Before (0.3.x)
let db = SolverDatabase::generate_from_hipparcos("data/hip2.dat", &config)?;

// After (0.4.0)
let db = SolverDatabase::generate_from_gaia("data/gaia_merged.bin", &config)?;
```

**Python:**
```python
# Before (0.3.x)
db = tetra3rs.SolverDatabase.generate_from_hipparcos()

# After (0.4.0)
db = tetra3rs.SolverDatabase.generate_from_gaia()  # uses bundled gaia-catalog
```

## 0.3.2

Initial public release with Hipparcos catalog support, centroid extraction, camera model, distortion calibration, WCS output, and stellar aberration correction.
