use std::env;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::{Path, PathBuf};
use std::slice;

use libloading::Library;

#[repr(C)]
#[derive(Clone, Copy)]
struct DataloaderSkipConfig {
    filtered: bool,
    random_fen_skipping: c_int,
    wld_filtered: bool,
    early_fen_skipping: c_int,
    simple_eval_skipping: c_int,
    param_index: c_int,
    pc_y1: f64,
    pc_y2: f64,
    pc_y3: f64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DataloaderDDPConfig {
    rank: c_int,
    world_size: c_int,
}

#[repr(C)]
struct SparseBatch {
    num_inputs: c_int,
    size: c_int,
    is_white: *const f32,
    outcome: *const f32,
    score: *const f32,
    num_active_white_features: c_int,
    num_active_black_features: c_int,
    max_active_features: c_int,
    white: *const c_int,
    black: *const c_int,
    white_values: *const f32,
    black_values: *const f32,
    psqt_indices: *const c_int,
    layer_stack_indices: *const c_int,
}

#[repr(C)]
struct Fen {
    size: c_int,
    fen: *const c_char,
}

#[repr(C)]
struct FenBatch {
    size: c_int,
    fens: *const Fen,
}

type GetSparseBatchFromFensFn = unsafe extern "C" fn(
    *const c_char,
    c_int,
    *const *const c_char,
    *mut c_int,
    *mut c_int,
    *mut c_int,
) -> *mut SparseBatch;
type DestroySparseBatchFn = unsafe extern "C" fn(*mut SparseBatch);
type CreateFenBatchStreamFn = unsafe extern "C" fn(
    c_int,
    c_int,
    *const *const c_char,
    c_int,
    bool,
    DataloaderSkipConfig,
    DataloaderDDPConfig,
) -> *mut c_void;
type DestroyFenBatchStreamFn = unsafe extern "C" fn(*mut c_void);
type FetchNextFenBatchFn = unsafe extern "C" fn(*mut c_void) -> *mut FenBatch;
type DestroyFenBatchFn = unsafe extern "C" fn(*mut FenBatch);

pub struct ReferenceBatch {
    pub num_inputs: i32,
    pub size: usize,
    pub max_active_features: usize,
    pub num_active_white_features: i32,
    pub num_active_black_features: i32,
    pub is_white: Vec<f32>,
    pub outcome: Vec<f32>,
    pub score: Vec<f32>,
    pub white: Vec<i32>,
    pub black: Vec<i32>,
    pub white_values: Vec<f32>,
    pub black_values: Vec<f32>,
    pub psqt_indices: Vec<i32>,
    pub layer_stack_indices: Vec<i32>,
}

impl ReferenceBatch {
    pub fn white_row(&self, row: usize) -> &[i32] {
        self.row_i32(&self.white, row)
    }

    pub fn black_row(&self, row: usize) -> &[i32] {
        self.row_i32(&self.black, row)
    }

    pub fn white_values_row(&self, row: usize) -> &[f32] {
        self.row_f32(&self.white_values, row)
    }

    pub fn black_values_row(&self, row: usize) -> &[f32] {
        self.row_f32(&self.black_values, row)
    }

    fn row_i32<'a>(&'a self, data: &'a [i32], row: usize) -> &'a [i32] {
        let start = row * self.max_active_features;
        &data[start..start + self.max_active_features]
    }

    fn row_f32<'a>(&'a self, data: &'a [f32], row: usize) -> &'a [f32] {
        let start = row * self.max_active_features;
        &data[start..start + self.max_active_features]
    }
}

pub struct CppReference {
    _library: Library,
    get_sparse_batch_from_fens: GetSparseBatchFromFensFn,
    destroy_sparse_batch: DestroySparseBatchFn,
    create_fen_batch_stream: CreateFenBatchStreamFn,
    destroy_fen_batch_stream: DestroyFenBatchStreamFn,
    fetch_next_fen_batch: FetchNextFenBatchFn,
    destroy_fen_batch: DestroyFenBatchFn,
}

impl CppReference {
    pub fn load() -> Result<Self, String> {
        let library_path = cpp_loader_path();
        let library = unsafe { Library::new(&library_path) }.map_err(|err| {
            format!(
                "failed to load C++ loader at {}: {err}",
                library_path.display()
            )
        })?;

        unsafe {
            let get_sparse_batch_from_fens = *library
                .get::<GetSparseBatchFromFensFn>(b"get_sparse_batch_from_fens")
                .map_err(|err| format!("failed to load get_sparse_batch_from_fens: {err}"))?;
            let destroy_sparse_batch = *library
                .get::<DestroySparseBatchFn>(b"destroy_sparse_batch")
                .map_err(|err| format!("failed to load destroy_sparse_batch: {err}"))?;
            let create_fen_batch_stream = *library
                .get::<CreateFenBatchStreamFn>(b"create_fen_batch_stream")
                .map_err(|err| format!("failed to load create_fen_batch_stream: {err}"))?;
            let destroy_fen_batch_stream = *library
                .get::<DestroyFenBatchStreamFn>(b"destroy_fen_batch_stream")
                .map_err(|err| format!("failed to load destroy_fen_batch_stream: {err}"))?;
            let fetch_next_fen_batch = *library
                .get::<FetchNextFenBatchFn>(b"fetch_next_fen_batch")
                .map_err(|err| format!("failed to load fetch_next_fen_batch: {err}"))?;
            let destroy_fen_batch = *library
                .get::<DestroyFenBatchFn>(b"destroy_fen_batch")
                .map_err(|err| format!("failed to load destroy_fen_batch: {err}"))?;

            Ok(Self {
                _library: library,
                get_sparse_batch_from_fens,
                destroy_sparse_batch,
                create_fen_batch_stream,
                destroy_fen_batch_stream,
                fetch_next_fen_batch,
                destroy_fen_batch,
            })
        }
    }

    pub fn batch_from_fens(
        &self,
        feature_set: &str,
        fens: &[String],
        scores: &[i32],
        plies: &[i32],
        results: &[i32],
    ) -> Result<ReferenceBatch, String> {
        if fens.is_empty() {
            return Err("cannot request an empty batch".to_string());
        }
        if fens.len() != scores.len() || fens.len() != plies.len() || fens.len() != results.len() {
            return Err("FEN, score, ply, and result lengths must match".to_string());
        }

        let feature_name = CString::new(feature_set)
            .map_err(|_| format!("feature set contains interior NUL: {feature_set}"))?;
        let fen_storage = fens
            .iter()
            .map(|fen| {
                CString::new(fen.as_str()).map_err(|_| format!("FEN contains interior NUL: {fen}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let fen_ptrs = fen_storage
            .iter()
            .map(|fen| fen.as_ptr())
            .collect::<Vec<_>>();
        let mut scores = scores.to_vec();
        let mut plies = plies.to_vec();
        let mut results = results.to_vec();

        let batch_ptr = unsafe {
            (self.get_sparse_batch_from_fens)(
                feature_name.as_ptr(),
                fens.len() as c_int,
                fen_ptrs.as_ptr(),
                scores.as_mut_ptr(),
                plies.as_mut_ptr(),
                results.as_mut_ptr(),
            )
        };
        if batch_ptr.is_null() {
            return Err("C++ loader returned a null SparseBatch pointer".to_string());
        }

        let batch = unsafe { &*batch_ptr };
        let size = batch.size as usize;
        let max_active_features = batch.max_active_features as usize;
        let flat_len = size * max_active_features;

        let snapshot = unsafe {
            ReferenceBatch {
                num_inputs: batch.num_inputs,
                size,
                max_active_features,
                num_active_white_features: batch.num_active_white_features,
                num_active_black_features: batch.num_active_black_features,
                is_white: slice::from_raw_parts(batch.is_white, size).to_vec(),
                outcome: slice::from_raw_parts(batch.outcome, size).to_vec(),
                score: slice::from_raw_parts(batch.score, size).to_vec(),
                white: slice::from_raw_parts(batch.white, flat_len).to_vec(),
                black: slice::from_raw_parts(batch.black, flat_len).to_vec(),
                white_values: slice::from_raw_parts(batch.white_values, flat_len).to_vec(),
                black_values: slice::from_raw_parts(batch.black_values, flat_len).to_vec(),
                psqt_indices: slice::from_raw_parts(batch.psqt_indices, size).to_vec(),
                layer_stack_indices: slice::from_raw_parts(batch.layer_stack_indices, size)
                    .to_vec(),
            }
        };

        unsafe {
            (self.destroy_sparse_batch)(batch_ptr);
        }

        Ok(snapshot)
    }

    pub fn sample_fens(
        &self,
        binpack_path: &Path,
        batch_size: usize,
    ) -> Result<Vec<String>, String> {
        let filename = CString::new(binpack_path.to_string_lossy().as_bytes()).map_err(|_| {
            format!(
                "binpack path contains interior NUL: {}",
                binpack_path.display()
            )
        })?;
        let filename_ptrs = [filename.as_ptr()];

        let config = DataloaderSkipConfig {
            filtered: false,
            random_fen_skipping: 0,
            wld_filtered: false,
            early_fen_skipping: -1,
            simple_eval_skipping: -1,
            param_index: 0,
            pc_y1: 1.0,
            pc_y2: 2.0,
            pc_y3: 1.0,
        };
        let ddp = DataloaderDDPConfig {
            rank: 0,
            world_size: 1,
        };

        let stream = unsafe {
            (self.create_fen_batch_stream)(
                1,
                1,
                filename_ptrs.as_ptr(),
                batch_size as c_int,
                false,
                config,
                ddp,
            )
        };
        if stream.is_null() {
            return Err("C++ loader returned a null FenBatchStream pointer".to_string());
        }

        let batch_ptr = unsafe { (self.fetch_next_fen_batch)(stream) };
        let result = if batch_ptr.is_null() {
            Err(format!(
                "no FEN batch returned for dataset {}",
                binpack_path.display()
            ))
        } else {
            let batch = unsafe { &*batch_ptr };
            let fens = unsafe {
                slice::from_raw_parts(batch.fens, batch.size as usize)
                    .iter()
                    .map(|fen| CStr::from_ptr(fen.fen).to_string_lossy().into_owned())
                    .collect::<Vec<_>>()
            };
            unsafe {
                (self.destroy_fen_batch)(batch_ptr);
            }
            Ok(fens)
        };

        unsafe {
            (self.destroy_fen_batch_stream)(stream);
        }

        result
    }
}

pub fn default_binpack_path() -> Option<PathBuf> {
    if let Some(path) = env::var_os("NNUE_PARITY_BINPACK") {
        return Some(PathBuf::from(path));
    }

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let default = manifest_dir.join("../nnue-data/nodes5000pv2_UHO.binpack");
    if default.exists() {
        return Some(default);
    }

    let data_dir = manifest_dir.join("../nnue-data");
    let entries = std::fs::read_dir(data_dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) == Some("binpack") {
            return Some(path);
        }
    }

    None
}

fn cpp_loader_path() -> PathBuf {
    if let Some(path) = env::var_os("NNUE_CPP_LOADER_LIB") {
        return PathBuf::from(path);
    }

    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../build/libtraining_data_loader.so")
}
