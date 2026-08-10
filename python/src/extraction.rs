use numpy::PyReadonlyArray2;
use pyo3::prelude::*;

use tetra3::centroid_extraction::{
    CentroidExtractionConfig, CentroidExtractionResult, DeblendMode, FastCentroidConfig,
};

use crate::centroid::PyCentroid;

/// The extraction result payload. Serde-derived, so it doubles as the pickle
/// wire format (layout unchanged from the former separate mirror struct).
#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct ExtractionData {
    centroids: Vec<tetra3::Centroid>,
    image_width: u32,
    image_height: u32,
    background_mean: f64,
    background_sigma: f64,
    threshold: f64,
    num_blobs_raw: u64,
}

impl From<CentroidExtractionResult> for ExtractionData {
    fn from(result: CentroidExtractionResult) -> Self {
        Self {
            centroids: result.centroids,
            image_width: result.image_width,
            image_height: result.image_height,
            background_mean: result.background_mean as f64,
            background_sigma: result.background_sigma as f64,
            threshold: result.threshold as f64,
            num_blobs_raw: result.num_blobs_raw as u64,
        }
    }
}

/// Convert a 2D numpy array of any supported dtype to Vec<f32>.
///
/// Supported dtypes: float64, float32, uint8, uint16, int16. Non-native byte
/// order (e.g. the big-endian `>f4` that `np.frombuffer` produces from FITS
/// data) is converted to native order first.
fn image_to_f32(image: &Bound<'_, pyo3::PyAny>) -> PyResult<(Vec<f32>, u32, u32)> {
    use pyo3::exceptions::PyTypeError;

    // Require a numpy array. A plain list (or anything without `dtype`) should
    // get a clear TypeError rather than a confusing AttributeError.
    let Ok(dtype) = image.getattr("dtype") else {
        return Err(PyTypeError::new_err(
            "image must be a 2D numpy array (got an object with no 'dtype')",
        ));
    };
    let ndim: usize = image.getattr("ndim")?.extract()?;
    if ndim != 2 {
        return Err(PyTypeError::new_err(format!(
            "image must be a 2D numpy array, got {ndim} dimension(s)"
        )));
    }

    // Normalize non-native byte order (FITS arrays are big-endian). `astype`
    // with a native-order dtype yields a correctly-valued native copy; recurse
    // once on it (it then passes the is-native check).
    let is_native: bool = dtype.getattr("isnative")?.extract()?;
    if !is_native {
        let native_dtype = dtype.call_method1("newbyteorder", ("=",))?;
        let converted = image.call_method1("astype", (native_dtype,))?;
        return image_to_f32(&converted);
    }

    let kind: String = dtype.getattr("kind")?.extract()?;
    let itemsize: usize = dtype.getattr("itemsize")?.extract()?;

    /// Extract one supported dtype and convert element-wise to f32.
    fn arr_to_f32<T: numpy::Element + Copy>(
        image: &Bound<'_, pyo3::PyAny>,
        to_f32: impl Fn(T) -> f32,
    ) -> PyResult<(Vec<f32>, u32, u32)> {
        let arr: PyReadonlyArray2<T> = image.extract()?;
        let a = arr.as_array();
        let h = a.shape()[0] as u32;
        let w = a.shape()[1] as u32;
        Ok((a.iter().map(|&v| to_f32(v)).collect(), w, h))
    }

    match (kind.as_str(), itemsize) {
        ("f", 8) => arr_to_f32::<f64>(image, |v| v as f32),
        ("f", 4) => arr_to_f32::<f32>(image, |v| v),
        ("u", 1) => arr_to_f32::<u8>(image, |v| v as f32),
        ("u", 2) => arr_to_f32::<u16>(image, |v| v as f32),
        ("i", 2) => arr_to_f32::<i16>(image, |v| v as f32),
        _ => {
            let dtype_str: String = dtype.str()?.extract()?;
            Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "Unsupported image dtype '{}'. Expected float64, float32, uint16, int16, or uint8.",
                dtype_str,
            )))
        }
    }
}

/// Result of centroid extraction from an image.
#[pyclass(name = "ExtractionResult", module = "tetra3rs", frozen)]
pub(crate) struct PyExtractionResult {
    inner: ExtractionData,
}

#[pymethods]
impl PyExtractionResult {
    /// List of detected centroids, sorted by brightness (brightest first).
    #[getter]
    fn centroids(&self) -> Vec<PyCentroid> {
        self.inner
            .centroids
            .iter()
            .map(|c| PyCentroid { inner: c.clone() })
            .collect()
    }

    /// Width of the input image in pixels.
    #[getter]
    fn image_width(&self) -> u32 {
        self.inner.image_width
    }

    /// Height of the input image in pixels.
    #[getter]
    fn image_height(&self) -> u32 {
        self.inner.image_height
    }

    /// Estimated background mean.
    #[getter]
    fn background_mean(&self) -> f64 {
        self.inner.background_mean
    }

    /// Estimated background standard deviation.
    #[getter]
    fn background_sigma(&self) -> f64 {
        self.inner.background_sigma
    }

    /// Detection threshold used.
    #[getter]
    fn threshold(&self) -> f64 {
        self.inner.threshold
    }

    /// Number of raw blobs before filtering.
    #[getter]
    fn num_blobs_raw(&self) -> usize {
        self.inner.num_blobs_raw as usize
    }

    fn __reduce__(slf: &Bound<'_, Self>) -> PyResult<(Py<PyAny>, (Vec<u8>,))> {
        crate::helpers::pickle_reduce(slf, &slf.borrow().inner)
    }

    #[staticmethod]
    fn _from_pickle_bytes(data: &[u8]) -> PyResult<Self> {
        let inner = crate::helpers::from_postcard_bytes::<ExtractionData>(data)?;
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "ExtractionResult(centroids={}, image={}x{}, bg_mean={:.1}, bg_sigma={:.1}, threshold={:.1}, raw_blobs={})",
            self.inner.centroids.len(),
            self.inner.image_width,
            self.inner.image_height,
            self.inner.background_mean,
            self.inner.background_sigma,
            self.inner.threshold,
            self.inner.num_blobs_raw,
        )
    }
}

/// Extract star centroids from a 2D image array.
///
/// Detects stars by sigma-clipping background estimation, thresholding, connected-
/// component labeling, and intensity-weighted centroiding. Each blob's background
/// is refined using the median of nearby non-blob pixels (annulus), and a 2D
/// quadratic fit to the 3×3 peak neighborhood provides sub-pixel precision.
///
/// Args:
///     image: 2D numpy array (height x width) of pixel values.
///         Supported dtypes: float64, float32, uint16, int16, uint8.
///     sigma_threshold: Detection threshold in sigma above background. Default 5.0.
///     min_pixels: Minimum blob size. Default 3.
///     max_pixels: Maximum blob size. Default 10000.
///     max_centroids: Maximum number of centroids to return. None = all.
///     local_bg_block_size: Block size for local background estimation. None = global only.
///     max_elongation: Maximum blob elongation ratio. None = disabled.
///     matched_filter_sigma: Apply a Gaussian matched filter of this sigma
///         (in pixels) before thresholding (~2x point-source SNR for a
///         sigma~1.5 px PSF). Used only to form the detection mask, so
///         photometry is unaffected, and the threshold is automatically
///         scaled for the filtered noise level — no retuning needed.
///         None = disabled. Default 1.5.
///     max_sharpness: Reject blobs whose peak sharpness
///         ``(peak - mean(8 neighbors)) / peak`` exceeds this — values near 1
///         are hot pixels / cosmic rays, not stars. A critically sampled PSF
///         scores ~0.5; strongly undersampled optics up to ~0.85. Set to
///         None for severely undersampled data (PSF FWHM below ~1.5 px),
///         where real stars are indistinguishable from hot pixels.
///         Default 0.9.
///     saturation_level: Pixel value at or above which the sensor is
///         saturated; such blobs skip sub-pixel peak refinement (a flat top
///         has no meaningful maximum) and keep the center-of-mass position.
///         None = disabled.
///     deblend: Policy for blobs with more than one distinct intensity peak
///         (blended star pairs centroid to a wrong midpoint position).
///         "off" keeps them merged; "reject" drops them — the safe choice
///         for plate solving. Saturated blobs are exempt. Default "off".
///     border_margin: Drop blobs whose bounding box comes within this many
///         pixels of an image edge (truncated PSFs bias the center-of-mass
///         inward). Default 0 (disabled).
///
/// Returns:
///     ExtractionResult with centroids and image statistics.
#[pyfunction]
#[pyo3(signature = (
    image,
    sigma_threshold = 5.0,
    min_pixels = 3,
    max_pixels = 10000,
    max_centroids = None,
    local_bg_block_size = Some(64),
    max_elongation = Some(3.0),
    matched_filter_sigma = Some(1.5),
    max_sharpness = Some(0.9),
    saturation_level = None,
    deblend = "off",
    border_margin = 0,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn extract_centroids(
    image: &Bound<'_, pyo3::PyAny>,
    sigma_threshold: f32,
    min_pixels: usize,
    max_pixels: usize,
    max_centroids: Option<usize>,
    local_bg_block_size: Option<u32>,
    max_elongation: Option<f32>,
    matched_filter_sigma: Option<f32>,
    max_sharpness: Option<f32>,
    saturation_level: Option<f32>,
    deblend: &str,
    border_margin: u32,
) -> PyResult<PyExtractionResult> {
    let (pixels, width, height) = image_to_f32(image)?;

    let deblend = match deblend {
        "off" => DeblendMode::Off,
        "reject" => DeblendMode::Reject,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "deblend must be 'off' or 'reject', got '{other}'"
            )))
        }
    };

    let config = CentroidExtractionConfig {
        sigma_threshold,
        min_pixels,
        max_pixels,
        max_centroids,
        sigma_clip_iterations: 5,
        sigma_clip_factor: 3.0,
        local_bg_block_size,
        max_elongation,
        matched_filter_sigma,
        max_sharpness,
        saturation_level,
        deblend,
        border_margin,
    };

    let result =
        tetra3::centroid_extraction::extract_centroids_from_raw(&pixels, width, height, &config)
            .map_err(crate::helpers::map_tetra3_err)?;

    Ok(PyExtractionResult {
        inner: result.into(),
    })
}

/// Fast single-pass centroid extraction — the "adequate star tracker" path.
///
/// Reads each pixel once: a cheap subsampled pre-pass builds a coarse
/// background grid, then a single raster sweep thresholds against the
/// interpolated background and groups lit pixels into connected regions
/// (run-length + union-find), emitting one center-of-mass per region. No
/// convolution and no second pass, so it is several times faster than
/// :func:`extract_centroids` — at the cost of faint-star sensitivity and
/// sub-pixel accuracy (~0.1 px on bright stars). Use :func:`extract_centroids`
/// for calibration or faint-star work.
///
/// Returns the same ``ExtractionResult`` as :func:`extract_centroids`, so it
/// is a drop-in for ``solve_from_centroids``.
///
/// Args:
///     image: 2D numpy array (height x width) of pixel values.
///         Supported dtypes: float64, float32, uint16, int16, uint8.
///     sigma_threshold: Detection threshold in noise sigmas above the local
///         background. Default 5.0.
///     bg_grid: Coarse background-grid block size in pixels. Gradients
///         (vignetting, Milky Way) are tracked via this grid. Default 64.
///     min_pixels: Minimum pixels in a region (rejects hot pixels). Default 2.
///     max_centroids: Maximum number of centroids to return, brightest first.
///         None = all. A few dozen is plenty for solving / tracking.
///     max_sharpness: Reject regions whose peak sharpness
///         ``(peak - mean(8 neighbors)) / peak`` exceeds this — values near 1
///         are hot pixels / cosmic rays, not stars. Set to None for
///         severely undersampled data (PSF FWHM below ~1.5 px).
///         Default 0.9.
///     saturation_level: Pixel value at or above which the sensor is
///         saturated; such regions skip sub-pixel peak refinement and keep
///         the center-of-mass position. None = disabled.
///     max_pixels: Maximum region size in pixels — without it a satellite
///         trail or horizon glow becomes the *brightest* centroid handed to
///         the solver. Default 10000.
///     max_elongation: Maximum elongation ratio (major/minor axis) from
///         intensity-weighted second moments — rejects streaks too small for
///         max_pixels. Off by default: moment-based elongation is noisy for
///         regions of a few pixels; enable (3.0-5.0) with min_pixels raised
///         to ~5+. None = disabled.
///     border_margin: Drop regions whose bounding box comes within this many
///         pixels of an image edge (truncated PSFs bias the center-of-mass
///         inward). Default 0 (disabled).
///
/// Returns:
///     ExtractionResult with centroids and image statistics.
#[pyfunction]
#[pyo3(signature = (
    image,
    sigma_threshold = 5.0,
    bg_grid = 64,
    min_pixels = 2,
    max_centroids = None,
    max_sharpness = Some(0.9),
    saturation_level = None,
    max_pixels = 10000,
    max_elongation = None,
    border_margin = 0,
))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn extract_centroids_fast(
    image: &Bound<'_, pyo3::PyAny>,
    sigma_threshold: f32,
    bg_grid: u32,
    min_pixels: usize,
    max_centroids: Option<usize>,
    max_sharpness: Option<f32>,
    saturation_level: Option<f32>,
    max_pixels: usize,
    max_elongation: Option<f32>,
    border_margin: u32,
) -> PyResult<PyExtractionResult> {
    let (pixels, width, height) = image_to_f32(image)?;

    let config = FastCentroidConfig {
        sigma_threshold,
        bg_grid,
        min_pixels,
        max_centroids,
        max_sharpness,
        saturation_level,
        max_pixels,
        max_elongation,
        border_margin,
    };

    let result =
        tetra3::centroid_extraction::extract_centroids_fast(&pixels, width, height, &config)
            .map_err(crate::helpers::map_tetra3_err)?;

    Ok(PyExtractionResult {
        inner: result.into(),
    })
}
