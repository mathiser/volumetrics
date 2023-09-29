pub mod utils;
pub mod overlap;
pub mod distance;
pub mod apl;
use ndarray::prelude::*;
use numpy::{PyArrayDyn, ToPyArray, PyReadonlyArrayDyn, PyReadonlyArray3, PyArray};
use Vec;
use pyo3::{
    pymodule,
    types::{PyModule},
    PyResult, Python
};

/// A Python module implemented in Rust.
#[pymodule]
fn volumetrics<'py>(_py: Python<'py>, m: &'py PyModule) -> PyResult<()> {
    #[pyfn(m)]
    #[pyo3(name = "echo_array")]
    fn echo_array<'py>(_py: Python<'py>, arr: PyReadonlyArrayDyn<'py, bool>) -> &'py PyArrayDyn<bool> {
        let arr = arr.as_array();
        arr.to_pyarray(_py)
    }
    #[pyfn(m)]
    #[pyo3(name = "generate_edge")]
    fn generate_edge<'py>(_py: Python<'py>, arr: PyReadonlyArray3<'py, bool>, use_2d: bool) -> &'py PyArray<bool, Ix3> {
        let arr = arr.to_owned_array();
        let edge_arr = crate::utils::generate_edge(&arr, use_2d);
        edge_arr.to_pyarray(_py)
    }
    #[pyfn(m)]
    #[pyo3(name = "surface_dice")]
    fn surface_dice<'py>(_py: Python<'py>,
                         ref_arr: PyReadonlyArray3<'py, bool>,
                         other_arr: PyReadonlyArray3<'py, bool>,
                         zyx_spacing: Vec<f32>,
                         tolerances: Vec<f32>) -> Vec<f32> {

        let ref_arr = ref_arr.to_owned_array();
        let other_arr = other_arr.to_owned_array();

        let mut hd_map = crate::distance::HausdorffMapDirected::new(&ref_arr, &other_arr, zyx_spacing);
        hd_map.execute();
        let mut sds = Vec::<f32>::new();
        for t in tolerances{
            sds.push(hd_map.surface_dc(&t))
        }
        sds
    }
    #[pyfn(m)]
    #[pyo3(name = "hd")]
    fn hd<'py>(_py: Python<'py>,
               ref_arr: PyReadonlyArray3<'py, bool>,
               other_arr: PyReadonlyArray3<'py, bool>,
               zyx_spacing: Vec<f32>,
               undirected: bool,
    ) -> f32 {
        let ref_arr = ref_arr.to_owned_array();
        let other_arr = other_arr.to_owned_array();

        if undirected {
            let mut hd_map = crate::distance::HausdorffMapUndirected::new(&ref_arr, &other_arr, zyx_spacing);
            hd_map.execute();
            hd_map.hd()
        } else {
            let mut hd_map = crate::distance::HausdorffMapDirected::new(&ref_arr, &other_arr, zyx_spacing);
            hd_map.execute();
            hd_map.hd()
        }
    }
    #[pyfn(m)]
    #[pyo3(name = "hd_percentile")]
    fn hd_percentile<'py>(_py: Python<'py>,
                          ref_arr: PyReadonlyArray3<'py, bool>,
                          other_arr: PyReadonlyArray3<'py, bool>,
                          zyx_spacing: Vec<f32>,
                          percentile: f32,
                          undirected: bool,
    ) -> f32 {
        let ref_arr = ref_arr.to_owned_array();
        let other_arr = other_arr.to_owned_array();

        if undirected {
            let mut hd_map = crate::distance::HausdorffMapUndirected::new(&ref_arr, &other_arr, zyx_spacing);
            hd_map.execute();
            hd_map.hd_percentile(&percentile)
        } else {
            let mut hd_map = crate::distance::HausdorffMapDirected::new(&ref_arr, &other_arr, zyx_spacing);
            hd_map.execute();
            hd_map.hd_percentile(&percentile)
        }
    }
    #[pyfn(m)]
    #[pyo3(name = "dc")]
    fn dc<'py>(_py: Python<'py>,
               ref_arr: PyReadonlyArray3<'py, bool>,
               other_arr: PyReadonlyArray3<'py, bool>,
    ) -> f32 {
        let ref_arr = ref_arr.to_owned_array();
        let other_arr = other_arr.to_owned_array();
        let mut cm = crate::overlap::ConfusionMatrix::new(&ref_arr, &other_arr);
        cm.execute();
        cm.dc()
    }
    #[pyfn(m)]
    #[pyo3(name = "jc")]
    fn jc<'py>(_py: Python<'py>,
               ref_arr: PyReadonlyArray3<'py, bool>,
               other_arr: PyReadonlyArray3<'py, bool>,
    ) -> f32 {
        let ref_arr = ref_arr.to_owned_array();
        let other_arr = other_arr.to_owned_array();
        let mut cm = crate::overlap::ConfusionMatrix::new(&ref_arr, &other_arr);
        cm.execute();
        cm.jc()
    }
    Ok(())

}