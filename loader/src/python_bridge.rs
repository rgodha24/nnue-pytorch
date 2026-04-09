use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Mutex;

use numpy::ndarray::{ArrayView1, ArrayView2};
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::{PyRuntimeError, PyStopIteration, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::feature_extraction::FeatureSet;
use crate::pipeline::{BatchPipeline, PipelineConfig, PipelineError, PooledBatch, SkipConfig};

#[pyclass(name = "BatchStream")]
pub struct PyBatchStream {
    pipeline: Mutex<Option<BatchPipeline>>,
}

#[pyclass]
struct PyBatchOwner {
    batch: PooledBatch,
    psqt_indices_i64: Vec<i64>,
    layer_stack_indices_i64: Vec<i64>,
}

#[pymethods]
impl PyBatchStream {
    #[new]
    #[pyo3(signature = (
        feature_set,
        filenames,
        batch_size,
        encoding_threads=None,
        slab_count=None,
        position_queue_capacity=None,
        position_queue_high_watermark=None,
        position_queue_low_watermark=None,
        shuffle_buffer_entries=1_000_000,
        shuffle_chunks=true,
        cyclic=false,
        seed=None,
        filtered=false,
        random_fen_skipping=0,
        wld_filtered=false,
        early_fen_skipping=-1,
        simple_eval_skipping=0,
        param_index=0,
        pc_y1=1.0,
        pc_y2=2.0,
        pc_y3=1.0
    ))]
    fn new(
        feature_set: &str,
        filenames: Vec<String>,
        batch_size: usize,
        encoding_threads: Option<usize>,
        slab_count: Option<usize>,
        position_queue_capacity: Option<usize>,
        position_queue_high_watermark: Option<usize>,
        position_queue_low_watermark: Option<usize>,
        shuffle_buffer_entries: usize,
        shuffle_chunks: bool,
        cyclic: bool,
        seed: Option<u64>,
        filtered: bool,
        random_fen_skipping: u32,
        wld_filtered: bool,
        early_fen_skipping: i32,
        simple_eval_skipping: i32,
        param_index: i32,
        pc_y1: f64,
        pc_y2: f64,
        pc_y3: f64,
    ) -> PyResult<Self> {
        Self::from_args(
            feature_set,
            filenames,
            batch_size,
            encoding_threads,
            slab_count,
            position_queue_capacity,
            position_queue_high_watermark,
            position_queue_low_watermark,
            shuffle_buffer_entries,
            shuffle_chunks,
            cyclic,
            seed,
            filtered,
            random_fen_skipping,
            wld_filtered,
            early_fen_skipping,
            simple_eval_skipping,
            param_index,
            pc_y1,
            pc_y2,
            pc_y3,
        )
    }

    fn next_batch<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let maybe_batch = py.allow_threads(|| {
            let guard = self.pipeline.lock().unwrap();
            match guard.as_ref() {
                Some(pipeline) => pipeline.next_batch(),
                None => Ok(None),
            }
        });

        match maybe_batch {
            Ok(Some(batch)) => Ok(Some(batch_to_pydict(py, batch)?)),
            Ok(None) => Ok(None),
            Err(error) => Err(to_py_runtime_error(error)),
        }
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        match self.next_batch(py)? {
            Some(batch) => Ok(batch),
            None => Err(PyStopIteration::new_err("batch stream is exhausted")),
        }
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let stats = {
            let guard = self.pipeline.lock().unwrap();
            match guard.as_ref() {
                Some(pipeline) => pipeline.stats(),
                None => {
                    return Err(PyRuntimeError::new_err(
                        "batch stream has already been closed",
                    ));
                }
            }
        };

        let dict = PyDict::new(py);
        dict.set_item("decoded_entries", stats.decoded_entries)?;
        dict.set_item("encoded_entries", stats.encoded_entries)?;
        dict.set_item("skipped_entries", stats.skipped_entries)?;
        dict.set_item("produced_batches", stats.produced_batches)?;
        dict.set_item("position_queue_len", stats.position_queue_len)?;
        dict.set_item("ready_slabs_len", stats.ready_slabs_len)?;
        dict.set_item("free_slabs_len", stats.free_slabs_len)?;
        Ok(dict)
    }

    fn close(&self) {
        let mut guard = self.pipeline.lock().unwrap();
        let _ = guard.take();
    }
}

#[pyfunction]
#[pyo3(signature = (
    feature_set,
    filenames,
    batch_size,
    encoding_threads=None,
    slab_count=None,
    position_queue_capacity=None,
    position_queue_high_watermark=None,
    position_queue_low_watermark=None,
    shuffle_buffer_entries=1_000_000,
    shuffle_chunks=true,
    cyclic=false,
    seed=None,
    filtered=false,
    random_fen_skipping=0,
    wld_filtered=false,
    early_fen_skipping=-1,
    simple_eval_skipping=0,
    param_index=0,
    pc_y1=1.0,
    pc_y2=2.0,
    pc_y3=1.0
))]
pub fn create_batch_stream(
    feature_set: &str,
    filenames: Vec<String>,
    batch_size: usize,
    encoding_threads: Option<usize>,
    slab_count: Option<usize>,
    position_queue_capacity: Option<usize>,
    position_queue_high_watermark: Option<usize>,
    position_queue_low_watermark: Option<usize>,
    shuffle_buffer_entries: usize,
    shuffle_chunks: bool,
    cyclic: bool,
    seed: Option<u64>,
    filtered: bool,
    random_fen_skipping: u32,
    wld_filtered: bool,
    early_fen_skipping: i32,
    simple_eval_skipping: i32,
    param_index: i32,
    pc_y1: f64,
    pc_y2: f64,
    pc_y3: f64,
) -> PyResult<PyBatchStream> {
    PyBatchStream::from_args(
        feature_set,
        filenames,
        batch_size,
        encoding_threads,
        slab_count,
        position_queue_capacity,
        position_queue_high_watermark,
        position_queue_low_watermark,
        shuffle_buffer_entries,
        shuffle_chunks,
        cyclic,
        seed,
        filtered,
        random_fen_skipping,
        wld_filtered,
        early_fen_skipping,
        simple_eval_skipping,
        param_index,
        pc_y1,
        pc_y2,
        pc_y3,
    )
}

impl PyBatchStream {
    #[allow(clippy::too_many_arguments)]
    fn from_args(
        feature_set: &str,
        filenames: Vec<String>,
        batch_size: usize,
        encoding_threads: Option<usize>,
        slab_count: Option<usize>,
        position_queue_capacity: Option<usize>,
        position_queue_high_watermark: Option<usize>,
        position_queue_low_watermark: Option<usize>,
        shuffle_buffer_entries: usize,
        shuffle_chunks: bool,
        cyclic: bool,
        seed: Option<u64>,
        filtered: bool,
        random_fen_skipping: u32,
        wld_filtered: bool,
        early_fen_skipping: i32,
        simple_eval_skipping: i32,
        param_index: i32,
        pc_y1: f64,
        pc_y2: f64,
        pc_y3: f64,
    ) -> PyResult<Self> {
        let feature_set = FeatureSet::from_str(feature_set)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let filenames = filenames.into_iter().map(PathBuf::from).collect();
        let mut config = PipelineConfig::new(filenames, feature_set, batch_size);

        if let Some(value) = encoding_threads {
            config.encoding_threads = value;
        }
        if let Some(value) = slab_count {
            config.slab_count = value;
        }
        if let Some(value) = position_queue_capacity {
            config.position_queue_capacity = value;
            if position_queue_high_watermark.is_none() {
                config.position_queue_high_watermark = value.saturating_mul(99) / 100;
            }
            if position_queue_low_watermark.is_none() {
                config.position_queue_low_watermark = value / 2;
            }
        }
        if let Some(value) = position_queue_high_watermark {
            config.position_queue_high_watermark = value;
        }
        if let Some(value) = position_queue_low_watermark {
            config.position_queue_low_watermark = value;
        }

        config.shuffle_buffer_entries = shuffle_buffer_entries;
        config.shuffle_chunks = shuffle_chunks;
        config.cyclic = cyclic;
        config.seed = seed;
        config.skip_config = SkipConfig {
            filtered,
            random_fen_skipping,
            wld_filtered,
            early_fen_skipping,
            simple_eval_skipping,
            param_index,
            pc_y1,
            pc_y2,
            pc_y3,
        };

        let pipeline = BatchPipeline::new(config).map_err(to_py_runtime_error)?;
        Ok(Self {
            pipeline: Mutex::new(Some(pipeline)),
        })
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBatchStream>()?;
    m.add_function(wrap_pyfunction!(create_batch_stream, m)?)?;
    Ok(())
}

fn batch_to_pydict<'py>(py: Python<'py>, batch: PooledBatch) -> PyResult<Bound<'py, PyDict>> {
    let owner = Py::new(py, PyBatchOwner::new(batch))?;
    let owner_bound = owner.bind(py);
    let owner_ref = owner_bound.borrow();
    let batch = owner_ref.batch.slab();
    let rows = batch.len();
    let cols = batch.max_active_features();

    let is_white_view = ArrayView2::from_shape((rows, 1), batch.is_white_slice())
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    let outcome_view = ArrayView2::from_shape((rows, 1), batch.outcome_slice())
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    let score_view = ArrayView2::from_shape((rows, 1), batch.score_slice())
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    let white_view = ArrayView2::from_shape((rows, cols), batch.white_flat_slice())
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    let black_view = ArrayView2::from_shape((rows, cols), batch.black_flat_slice())
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
    let psqt_indices_view = ArrayView1::from(owner_ref.psqt_indices_i64.as_slice());
    let layer_stack_indices_view = ArrayView1::from(owner_ref.layer_stack_indices_i64.as_slice());

    let is_white =
        unsafe { PyArray2::borrow_from_array(&is_white_view, owner_bound.clone().into_any()) };
    let outcome =
        unsafe { PyArray2::borrow_from_array(&outcome_view, owner_bound.clone().into_any()) };
    let score = unsafe { PyArray2::borrow_from_array(&score_view, owner_bound.clone().into_any()) };
    let white = unsafe { PyArray2::borrow_from_array(&white_view, owner_bound.clone().into_any()) };
    let black = unsafe { PyArray2::borrow_from_array(&black_view, owner_bound.clone().into_any()) };
    let psqt_indices =
        unsafe { PyArray1::borrow_from_array(&psqt_indices_view, owner_bound.clone().into_any()) };
    let layer_stack_indices = unsafe {
        PyArray1::borrow_from_array(&layer_stack_indices_view, owner_bound.clone().into_any())
    };

    let dict = PyDict::new(py);
    dict.set_item("num_inputs", batch.num_inputs())?;
    dict.set_item("size", rows)?;
    dict.set_item(
        "num_active_white_features",
        batch.num_active_white_features(),
    )?;
    dict.set_item(
        "num_active_black_features",
        batch.num_active_black_features(),
    )?;
    dict.set_item("max_active_features", cols)?;
    dict.set_item("is_white", is_white)?;
    dict.set_item("outcome", outcome)?;
    dict.set_item("score", score)?;
    dict.set_item("white", white)?;
    dict.set_item("black", black)?;
    dict.set_item("psqt_indices", psqt_indices)?;
    dict.set_item("layer_stack_indices", layer_stack_indices)?;
    Ok(dict)
}

fn to_py_runtime_error(error: PipelineError) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

impl PyBatchOwner {
    fn new(batch: PooledBatch) -> Self {
        let psqt_indices_i64 = batch
            .psqt_indices_slice()
            .iter()
            .map(|&value| value as i64)
            .collect();
        let layer_stack_indices_i64 = batch
            .layer_stack_indices_slice()
            .iter()
            .map(|&value| value as i64)
            .collect();

        Self {
            batch,
            psqt_indices_i64,
            layer_stack_indices_i64,
        }
    }
}
