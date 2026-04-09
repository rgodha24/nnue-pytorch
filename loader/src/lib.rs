use pyo3::prelude::*;

pub mod feature_extraction;
pub mod pipeline;
mod python_bridge;
pub use sfbinpack::chess;

/// Test function to verify the module is wired correctly
#[pyfunction]
fn hello() -> String {
    "hello from nnue_loader!".to_string()
}

#[pymodule]
fn _internal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    python_bridge::register(m)?;
    Ok(())
}
