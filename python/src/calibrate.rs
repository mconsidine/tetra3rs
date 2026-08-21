use pyo3::prelude::*;

use tetra3::CalibrateResult;

use crate::camera_model::PyCameraModel;

/// Result of camera calibration.
///
/// Returned by ``SolverDatabase.calibrate_camera``.
///
/// Attributes:
///     camera_model: The fitted CameraModel (focal length, crpix, distortion).
///     rmse_before_px: RMS residual in pixels before calibration.
///     rmse_after_px: RMS residual in pixels after calibration.
///     n_inliers: Number of inlier star matches used in the fit.
///     n_outliers: Number of outlier star matches rejected by sigma clipping.
///     iterations: Number of sigma-clip iterations performed.
#[pyclass(name = "CalibrateResult", module = "tetra3rs", frozen)]
pub(crate) struct PyCalibrateResult {
    pub(crate) inner: CalibrateResult,
}

#[pymethods]
impl PyCalibrateResult {
    /// The fitted CameraModel (focal length, crpix, distortion).
    #[getter]
    fn camera_model(&self) -> PyCameraModel {
        PyCameraModel {
            inner: self.inner.camera_model.clone(),
        }
    }

    /// RMS residual in pixels before calibration.
    #[getter]
    fn rmse_before_px(&self) -> f64 {
        self.inner.rmse_before_px
    }

    /// RMS residual in pixels after calibration.
    #[getter]
    fn rmse_after_px(&self) -> f64 {
        self.inner.rmse_after_px
    }

    /// Number of inlier star matches used in the fit.
    #[getter]
    fn n_inliers(&self) -> usize {
        self.inner.n_inliers
    }

    /// Number of outlier star matches rejected by sigma clipping.
    #[getter]
    fn n_outliers(&self) -> usize {
        self.inner.n_outliers
    }

    /// Number of sigma-clip iterations performed.
    #[getter]
    fn iterations(&self) -> u32 {
        self.inner.iterations
    }

    fn __reduce__(slf: &Bound<'_, Self>) -> PyResult<(Py<PyAny>, (Vec<u8>,))> {
        crate::helpers::pickle_reduce(slf, &slf.borrow().inner)
    }

    #[staticmethod]
    fn _from_pickle_bytes(data: &[u8]) -> PyResult<Self> {
        let inner = crate::helpers::from_postcard_bytes::<CalibrateResult>(data)?;
        // The embedded camera model's distortion is evaluated on use;
        // enforce its invariants against tampered pickle bytes.
        inner
            .camera_model
            .validate()
            .map_err(crate::helpers::map_tetra3_err)?;
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "CalibrateResult(rmse={:.3}->{:.3} px, inliers={}, outliers={}, iterations={})",
            self.inner.rmse_before_px,
            self.inner.rmse_after_px,
            self.inner.n_inliers,
            self.inner.n_outliers,
            self.inner.iterations,
        )
    }
}
