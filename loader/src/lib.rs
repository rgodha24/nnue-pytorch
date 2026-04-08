use pyo3::prelude::*;

pub mod feature_extraction;
pub use sfbinpack::chess;

/// Test function to verify the module is wired correctly
#[pyfunction]
fn hello() -> String {
    "hello from nnue_loader!".to_string()
}

#[pymodule]
fn _internal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    Ok(())
}
