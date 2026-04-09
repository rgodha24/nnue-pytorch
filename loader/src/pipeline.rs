use std::collections::VecDeque;
use std::fs::File;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

use crossbeam_queue::ArrayQueue;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use sfbinpack::chess::{color::Color, piece::Piece, piecetype::PieceType};
use sfbinpack::{CompressedReaderError, CompressedTrainingDataEntryReader, TrainingDataEntry};

use crate::feature_extraction::{
    encode_training_entry_indices_only, FeatureSet, RowMetadata, SparseRow,
};

const VALUE_NONE: i16 = 32002;
const MAX_SKIP_RATE: f64 = 10.0;

#[derive(Clone, Debug)]
pub struct PipelineConfig {
    pub files: Vec<PathBuf>,
    pub feature_set: FeatureSet,
    pub batch_size: usize,
    pub encoding_threads: usize,
    pub slab_count: usize,
    pub encoder_chunk_size: usize,
    pub chunk_queue_capacity: usize,
    pub position_queue_capacity: usize,
    pub position_queue_high_watermark: usize,
    pub position_queue_low_watermark: usize,
    pub shuffle_buffer_entries: usize,
    pub shuffle_chunks: bool,
    pub cyclic: bool,
    pub skip_config: SkipConfig,
    pub seed: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SkipConfig {
    pub filtered: bool,
    pub random_fen_skipping: u32,
    pub wld_filtered: bool,
    pub early_fen_skipping: i32,
    pub simple_eval_skipping: i32,
    pub param_index: i32,
    pub pc_y1: f64,
    pub pc_y2: f64,
    pub pc_y3: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PipelineStats {
    pub decoded_entries: u64,
    pub encoded_entries: u64,
    pub skipped_entries: u64,
    pub produced_batches: u64,
    pub position_queue_len: usize,
    pub ready_chunks_len: usize,
    pub free_chunks_len: usize,
    pub ready_slabs_len: usize,
    pub free_slabs_len: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PipelineError {
    message: String,
}

pub struct BatchPipeline {
    stop: Arc<AtomicBool>,
    ready_slabs: Arc<BlockingArrayQueue<HostBatchSlab>>,
    free_slabs: Arc<BlockingArrayQueue<HostBatchSlab>>,
    ready_chunks: Arc<BlockingArrayQueue<HostBatchSlab>>,
    free_chunks: Arc<BlockingArrayQueue<HostBatchSlab>>,
    position_queue: Arc<BlockingArrayQueue<TrainingDataEntry>>,
    error: Arc<Mutex<Option<PipelineError>>>,
    stats: Arc<PipelineCounters>,
    workers: Vec<JoinHandle<()>>,
}

pub struct PooledBatch {
    slab: Option<HostBatchSlab>,
    free_slabs: Arc<BlockingArrayQueue<HostBatchSlab>>,
    stop: Arc<AtomicBool>,
}

#[derive(Debug)]
pub struct HostBatchSlab {
    num_inputs: usize,
    max_active_features: usize,
    capacity: usize,
    size: usize,
    num_active_white_features: usize,
    num_active_black_features: usize,
    white_counts: Vec<u16>,
    black_counts: Vec<u16>,
    is_white: Vec<f32>,
    outcome: Vec<f32>,
    score: Vec<f32>,
    white: Vec<i32>,
    black: Vec<i32>,
    psqt_indices: Vec<i32>,
    layer_stack_indices: Vec<i32>,
}

struct BlockingArrayQueue<T> {
    queue: ArrayQueue<T>,
    wait_mutex: Mutex<()>,
    wait_cv: Condvar,
    closed: AtomicBool,
}

struct PipelineCounters {
    decoded_entries: AtomicU64,
    encoded_entries: AtomicU64,
    skipped_entries: AtomicU64,
    produced_batches: AtomicU64,
}

#[derive(Clone)]
struct ThreadContext {
    stop: Arc<AtomicBool>,
    error: Arc<Mutex<Option<PipelineError>>>,
    position_queue: Arc<BlockingArrayQueue<TrainingDataEntry>>,
    ready_chunks: Arc<BlockingArrayQueue<HostBatchSlab>>,
    free_chunks: Arc<BlockingArrayQueue<HostBatchSlab>>,
    ready_slabs: Arc<BlockingArrayQueue<HostBatchSlab>>,
    free_slabs: Arc<BlockingArrayQueue<HostBatchSlab>>,
    stats: Arc<PipelineCounters>,
}

struct SkipDecider {
    enabled: bool,
    config: SkipConfig,
    random_skip_probability: f64,
    desired_piece_count_weights_total: f64,
    alpha: f64,
    piece_count_history_all: [f64; 33],
    piece_count_history_passed: [f64; 33],
    piece_count_history_all_total: f64,
    piece_count_history_passed_total: f64,
}

impl PipelineConfig {
    pub fn new(files: Vec<PathBuf>, feature_set: FeatureSet, batch_size: usize) -> Self {
        let available_threads = thread::available_parallelism()
            .map(|threads| threads.get())
            .unwrap_or(1);
        let encoding_threads = available_threads.saturating_sub(files.len()).max(1);
        let encoder_chunk_size = default_encoder_chunk_size(batch_size, encoding_threads);
        let position_queue_capacity = 1 << 18;

        Self {
            files,
            feature_set,
            batch_size,
            encoding_threads,
            slab_count: default_batch_slab_count(encoding_threads),
            encoder_chunk_size,
            chunk_queue_capacity: default_chunk_queue_capacity(encoding_threads),
            position_queue_capacity,
            position_queue_high_watermark: position_queue_capacity * 99 / 100,
            position_queue_low_watermark: position_queue_capacity / 2,
            shuffle_buffer_entries: 1_000_000,
            shuffle_chunks: true,
            cyclic: false,
            skip_config: SkipConfig::default(),
            seed: None,
        }
    }

    fn validate(&self) -> Result<(), PipelineError> {
        if self.files.is_empty() {
            return Err(PipelineError::new(
                "pipeline requires at least one input file",
            ));
        }
        if self.batch_size == 0 {
            return Err(PipelineError::new("batch_size must be greater than zero"));
        }
        if self.encoding_threads == 0 {
            return Err(PipelineError::new(
                "encoding_threads must be greater than zero",
            ));
        }
        if self.slab_count == 0 {
            return Err(PipelineError::new("slab_count must be greater than zero"));
        }
        if self.encoder_chunk_size == 0 {
            return Err(PipelineError::new(
                "encoder_chunk_size must be greater than zero",
            ));
        }
        if self.encoder_chunk_size > self.batch_size {
            return Err(PipelineError::new(
                "encoder_chunk_size cannot exceed batch_size",
            ));
        }
        if self.chunk_queue_capacity == 0 {
            return Err(PipelineError::new(
                "chunk_queue_capacity must be greater than zero",
            ));
        }
        if self.position_queue_capacity < 2 {
            return Err(PipelineError::new(
                "position_queue_capacity must be at least 2 entries",
            ));
        }
        if self.position_queue_high_watermark == 0
            || self.position_queue_high_watermark > self.position_queue_capacity
        {
            return Err(PipelineError::new(
                "position_queue_high_watermark must be within queue capacity",
            ));
        }
        if self.position_queue_low_watermark >= self.position_queue_high_watermark {
            return Err(PipelineError::new(
                "position_queue_low_watermark must be lower than the high watermark",
            ));
        }
        if self.shuffle_buffer_entries == 0 {
            return Err(PipelineError::new(
                "shuffle_buffer_entries must be greater than zero",
            ));
        }

        Ok(())
    }
}

impl Default for SkipConfig {
    fn default() -> Self {
        Self {
            filtered: false,
            random_fen_skipping: 0,
            wld_filtered: false,
            early_fen_skipping: -1,
            simple_eval_skipping: 0,
            param_index: 0,
            pc_y1: 1.0,
            pc_y2: 2.0,
            pc_y3: 1.0,
        }
    }
}

impl BatchPipeline {
    pub fn new(config: PipelineConfig) -> Result<Self, PipelineError> {
        config.validate()?;

        let stop = Arc::new(AtomicBool::new(false));
        let error = Arc::new(Mutex::new(None));
        let stats = Arc::new(PipelineCounters::default());
        let position_queue = Arc::new(BlockingArrayQueue::new(config.position_queue_capacity));
        let ready_chunks = Arc::new(BlockingArrayQueue::new(config.chunk_queue_capacity));
        let free_chunks = Arc::new(BlockingArrayQueue::new(config.chunk_queue_capacity));
        let ready_slabs = Arc::new(BlockingArrayQueue::new(config.slab_count));
        let free_slabs = Arc::new(BlockingArrayQueue::new(config.slab_count));

        for _ in 0..config.chunk_queue_capacity {
            let chunk = HostBatchSlab::new(config.encoder_chunk_size, &config.feature_set);
            let _ = free_chunks.push_blocking(chunk, &stop);
        }

        for _ in 0..config.slab_count {
            let slab = HostBatchSlab::new(config.batch_size, &config.feature_set);
            let _ = free_slabs.push_blocking(slab, &stop);
        }

        let context = ThreadContext {
            stop: Arc::clone(&stop),
            error: Arc::clone(&error),
            position_queue: Arc::clone(&position_queue),
            ready_chunks: Arc::clone(&ready_chunks),
            free_chunks: Arc::clone(&free_chunks),
            ready_slabs: Arc::clone(&ready_slabs),
            free_slabs: Arc::clone(&free_slabs),
            stats: Arc::clone(&stats),
        };

        let mut workers = Vec::with_capacity(config.files.len() + config.encoding_threads + 1);
        let active_decoders = Arc::new(AtomicUsize::new(config.files.len()));
        let active_encoders = Arc::new(AtomicUsize::new(config.encoding_threads));

        for (index, path) in config.files.iter().cloned().enumerate() {
            let context = context.clone();
            let config = config.clone();
            let active_decoders = Arc::clone(&active_decoders);
            workers.push(thread::spawn(move || {
                decoder_worker(index, path, config, context, active_decoders)
            }));
        }

        for index in 0..config.encoding_threads {
            let context = context.clone();
            let config = config.clone();
            let active_encoders = Arc::clone(&active_encoders);
            workers.push(thread::spawn(move || {
                encoder_worker(index, config, context, active_encoders)
            }));
        }

        {
            let context = context.clone();
            let config = config.clone();
            workers.push(thread::spawn(move || assembler_worker(config, context)));
        }

        Ok(Self {
            stop,
            ready_slabs,
            free_slabs,
            ready_chunks,
            free_chunks,
            position_queue,
            error,
            stats,
            workers,
        })
    }

    pub fn next_batch(&self) -> Result<Option<PooledBatch>, PipelineError> {
        match self.ready_slabs.pop_blocking(&self.stop) {
            Some(slab) => Ok(Some(PooledBatch {
                slab: Some(slab),
                free_slabs: Arc::clone(&self.free_slabs),
                stop: Arc::clone(&self.stop),
            })),
            None => {
                if let Some(error) = self.error.lock().unwrap().clone() {
                    Err(error)
                } else {
                    Ok(None)
                }
            }
        }
    }

    pub fn stats(&self) -> PipelineStats {
        PipelineStats {
            decoded_entries: self.stats.decoded_entries.load(Ordering::Acquire),
            encoded_entries: self.stats.encoded_entries.load(Ordering::Acquire),
            skipped_entries: self.stats.skipped_entries.load(Ordering::Acquire),
            produced_batches: self.stats.produced_batches.load(Ordering::Acquire),
            position_queue_len: self.position_queue.len(),
            ready_chunks_len: self.ready_chunks.len(),
            free_chunks_len: self.free_chunks.len(),
            ready_slabs_len: self.ready_slabs.len(),
            free_slabs_len: self.free_slabs.len(),
        }
    }
}

impl Drop for BatchPipeline {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        self.position_queue.close();
        self.ready_chunks.close();
        self.free_chunks.close();
        self.ready_slabs.close();
        self.free_slabs.close();

        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

impl PooledBatch {
    pub fn slab(&self) -> &HostBatchSlab {
        self.slab
            .as_ref()
            .expect("pooled batch always holds a slab")
    }
}

impl std::ops::Deref for PooledBatch {
    type Target = HostBatchSlab;

    fn deref(&self) -> &Self::Target {
        self.slab()
    }
}

impl Drop for PooledBatch {
    fn drop(&mut self) {
        if let Some(mut slab) = self.slab.take() {
            slab.reset();
            let _ = self.free_slabs.push_blocking(slab, &self.stop);
        }
    }
}

impl HostBatchSlab {
    pub fn new(batch_size: usize, feature_set: &FeatureSet) -> Self {
        let max_active_features = feature_set.max_active_features();
        let flat_size = batch_size * max_active_features;

        Self {
            num_inputs: feature_set.inputs(),
            max_active_features,
            capacity: batch_size,
            size: 0,
            num_active_white_features: 0,
            num_active_black_features: 0,
            white_counts: vec![0; batch_size],
            black_counts: vec![0; batch_size],
            is_white: vec![0.0; batch_size],
            outcome: vec![0.0; batch_size],
            score: vec![0.0; batch_size],
            white: vec![-1; flat_size],
            black: vec![-1; flat_size],
            psqt_indices: vec![0; batch_size],
            layer_stack_indices: vec![0; batch_size],
        }
    }

    pub fn num_inputs(&self) -> usize {
        self.num_inputs
    }

    pub fn max_active_features(&self) -> usize {
        self.max_active_features
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    pub fn is_full(&self) -> bool {
        self.size == self.capacity
    }

    pub fn num_active_white_features(&self) -> usize {
        self.num_active_white_features
    }

    pub fn num_active_black_features(&self) -> usize {
        self.num_active_black_features
    }

    pub fn is_white_slice(&self) -> &[f32] {
        &self.is_white[..self.size]
    }

    pub fn outcome_slice(&self) -> &[f32] {
        &self.outcome[..self.size]
    }

    pub fn score_slice(&self) -> &[f32] {
        &self.score[..self.size]
    }

    pub fn psqt_indices_slice(&self) -> &[i32] {
        &self.psqt_indices[..self.size]
    }

    pub fn layer_stack_indices_slice(&self) -> &[i32] {
        &self.layer_stack_indices[..self.size]
    }

    pub fn white_row(&self, row: usize) -> &[i32] {
        let start = row * self.max_active_features;
        &self.white[start..start + self.max_active_features]
    }

    pub fn white_flat_slice(&self) -> &[i32] {
        &self.white[..self.size * self.max_active_features]
    }

    pub fn black_row(&self, row: usize) -> &[i32] {
        let start = row * self.max_active_features;
        &self.black[start..start + self.max_active_features]
    }

    pub fn black_flat_slice(&self) -> &[i32] {
        &self.black[..self.size * self.max_active_features]
    }

    pub fn copy_row(&self, row: usize) -> SparseRow {
        let white = self.white_row(row).to_vec();
        let black = self.black_row(row).to_vec();

        SparseRow {
            num_inputs: self.num_inputs,
            max_active_features: self.max_active_features,
            is_white: self.is_white[row],
            outcome: self.outcome[row],
            score: self.score[row],
            white_count: self.white_counts[row] as usize,
            black_count: self.black_counts[row] as usize,
            white_values: feature_values_from_indices(&white),
            black_values: feature_values_from_indices(&black),
            white,
            black,
            psqt_indices: self.psqt_indices[row],
            layer_stack_indices: self.layer_stack_indices[row],
        }
    }

    fn push_entry(&mut self, entry: &TrainingDataEntry, feature_set: &FeatureSet) {
        let row = self.size;
        let start = row * self.max_active_features;
        let end = start + self.max_active_features;
        let metadata = encode_training_entry_indices_only(
            entry,
            feature_set,
            &mut self.white[start..end],
            &mut self.black[start..end],
        );

        self.write_metadata(row, metadata);
        self.size += 1;
    }

    fn remaining_capacity(&self) -> usize {
        self.capacity - self.size
    }

    fn append_rows_from(&mut self, src: &HostBatchSlab, src_row: usize) -> usize {
        let rows = self
            .remaining_capacity()
            .min(src.len().saturating_sub(src_row));
        if rows == 0 {
            return 0;
        }

        let dst_row = self.size;
        let dst_end = dst_row + rows;
        let src_end = src_row + rows;
        let flat_width = self.max_active_features;
        let dst_flat_start = dst_row * flat_width;
        let dst_flat_end = dst_end * flat_width;
        let src_flat_start = src_row * flat_width;
        let src_flat_end = src_end * flat_width;

        self.white[dst_flat_start..dst_flat_end]
            .copy_from_slice(&src.white[src_flat_start..src_flat_end]);
        self.black[dst_flat_start..dst_flat_end]
            .copy_from_slice(&src.black[src_flat_start..src_flat_end]);
        self.white_counts[dst_row..dst_end].copy_from_slice(&src.white_counts[src_row..src_end]);
        self.black_counts[dst_row..dst_end].copy_from_slice(&src.black_counts[src_row..src_end]);
        self.is_white[dst_row..dst_end].copy_from_slice(&src.is_white[src_row..src_end]);
        self.outcome[dst_row..dst_end].copy_from_slice(&src.outcome[src_row..src_end]);
        self.score[dst_row..dst_end].copy_from_slice(&src.score[src_row..src_end]);
        self.psqt_indices[dst_row..dst_end].copy_from_slice(&src.psqt_indices[src_row..src_end]);
        self.layer_stack_indices[dst_row..dst_end]
            .copy_from_slice(&src.layer_stack_indices[src_row..src_end]);

        self.num_active_white_features += src.white_counts[src_row..src_end]
            .iter()
            .map(|&count| count as usize)
            .sum::<usize>();
        self.num_active_black_features += src.black_counts[src_row..src_end]
            .iter()
            .map(|&count| count as usize)
            .sum::<usize>();
        self.size = dst_end;

        rows
    }

    fn reset(&mut self) {
        self.size = 0;
        self.num_active_white_features = 0;
        self.num_active_black_features = 0;
    }

    fn write_metadata(&mut self, row: usize, metadata: RowMetadata) {
        self.white_counts[row] = metadata.white_count as u16;
        self.black_counts[row] = metadata.black_count as u16;
        self.is_white[row] = metadata.is_white;
        self.outcome[row] = metadata.outcome;
        self.score[row] = metadata.score;
        self.psqt_indices[row] = metadata.psqt_indices;
        self.layer_stack_indices[row] = metadata.layer_stack_indices;
        self.num_active_white_features += metadata.white_count;
        self.num_active_black_features += metadata.black_count;
    }
}

impl<T> BlockingArrayQueue<T> {
    fn new(capacity: usize) -> Self {
        Self {
            queue: ArrayQueue::new(capacity),
            wait_mutex: Mutex::new(()),
            wait_cv: Condvar::new(),
            closed: AtomicBool::new(false),
        }
    }

    fn len(&self) -> usize {
        self.queue.len()
    }

    fn close(&self) {
        self.closed.store(true, Ordering::Release);
        self.wait_cv.notify_all();
    }

    fn pop_blocking(&self, stop: &AtomicBool) -> Option<T> {
        loop {
            if let Some(item) = self.queue.pop() {
                self.wait_cv.notify_all();
                return Some(item);
            }

            if self.closed.load(Ordering::Acquire) || stop.load(Ordering::Acquire) {
                return None;
            }

            let mut guard = self.wait_mutex.lock().unwrap();
            while self.queue.is_empty()
                && !self.closed.load(Ordering::Acquire)
                && !stop.load(Ordering::Acquire)
            {
                guard = self.wait_cv.wait(guard).unwrap();
            }
        }
    }

    fn push_blocking(&self, mut item: T, stop: &AtomicBool) -> Result<(), T> {
        loop {
            match self.queue.push(item) {
                Ok(()) => {
                    self.wait_cv.notify_all();
                    return Ok(());
                }
                Err(returned) => item = returned,
            }

            if self.closed.load(Ordering::Acquire) || stop.load(Ordering::Acquire) {
                return Err(item);
            }

            let mut guard = self.wait_mutex.lock().unwrap();
            while self.queue.is_full()
                && !self.closed.load(Ordering::Acquire)
                && !stop.load(Ordering::Acquire)
            {
                guard = self.wait_cv.wait(guard).unwrap();
            }
        }
    }
}

impl Default for PipelineCounters {
    fn default() -> Self {
        Self {
            decoded_entries: AtomicU64::new(0),
            encoded_entries: AtomicU64::new(0),
            skipped_entries: AtomicU64::new(0),
            produced_batches: AtomicU64::new(0),
        }
    }
}

impl PipelineError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for PipelineError {}

impl SkipDecider {
    fn new(config: SkipConfig) -> Self {
        let enabled = config.filtered
            || config.random_fen_skipping > 0
            || config.wld_filtered
            || config.early_fen_skipping >= 0;
        let random_skip_probability = if config.random_fen_skipping == 0 {
            0.0
        } else {
            config.random_fen_skipping as f64 / (config.random_fen_skipping as f64 + 1.0)
        };

        let desired_piece_count_weights_total = (0..=32)
            .map(|piece_count| desired_piece_count_weight(config, piece_count))
            .sum();

        Self {
            enabled,
            config,
            random_skip_probability,
            desired_piece_count_weights_total,
            alpha: 1.0,
            piece_count_history_all: [0.0; 33],
            piece_count_history_passed: [0.0; 33],
            piece_count_history_all_total: 0.0,
            piece_count_history_passed_total: 0.0,
        }
    }

    fn should_skip(&mut self, entry: &TrainingDataEntry, rng: &mut SmallRng) -> bool {
        if !self.enabled {
            return false;
        }

        if entry.score == VALUE_NONE {
            return true;
        }
        if entry.ply as i32 <= self.config.early_fen_skipping {
            return true;
        }
        if self.config.random_fen_skipping > 0 && rng.gen_bool(self.random_skip_probability) {
            return true;
        }
        if self.config.filtered && (is_capturing_move(entry) || is_in_check(entry)) {
            return true;
        }
        if self.config.wld_filtered
            && rng.gen_bool((1.0 - score_result_prob(entry)).clamp(0.0, 1.0))
        {
            return true;
        }
        if self.config.simple_eval_skipping > 0
            && simple_eval(&entry.pos).abs() < self.config.simple_eval_skipping
        {
            return true;
        }

        let piece_count = entry.pos.occupied().count() as usize;
        self.piece_count_history_all[piece_count] += 1.0;
        self.piece_count_history_all_total += 1.0;

        if (self.piece_count_history_all_total as u64).is_multiple_of(10_000) {
            let current_weight = desired_piece_count_weight(self.config, piece_count as i32);
            let mut pass =
                self.piece_count_history_all_total * self.desired_piece_count_weights_total;

            for _ in 0..=32 {
                if current_weight > 0.0 {
                    let tmp = self.piece_count_history_all_total * current_weight
                        / (self.desired_piece_count_weights_total
                            * self.piece_count_history_all[piece_count]);
                    if tmp < pass {
                        pass = tmp;
                    }
                }
            }

            self.alpha = 1.0 / (pass * MAX_SKIP_RATE);
        }

        let tmp = (self.alpha
            * self.piece_count_history_all_total
            * desired_piece_count_weight(self.config, piece_count as i32)
            / (self.desired_piece_count_weights_total * self.piece_count_history_all[piece_count]))
            .min(1.0);

        if rng.gen_bool((1.0 - tmp).clamp(0.0, 1.0)) {
            return true;
        }

        self.piece_count_history_passed[piece_count] += 1.0;
        self.piece_count_history_passed_total += 1.0;
        false
    }
}

fn decoder_worker(
    worker_index: usize,
    path: PathBuf,
    config: PipelineConfig,
    context: ThreadContext,
    active_decoders: Arc<AtomicUsize>,
) {
    let base_seed = config.seed.unwrap_or_else(random_seed);
    let mut rng = SmallRng::seed_from_u64(base_seed ^ worker_index as u64);
    let mut skip_decider = SkipDecider::new(config.skip_config);
    let mut shuffled_entries = Vec::with_capacity(config.shuffle_buffer_entries);
    let mut ordered_entries = VecDeque::with_capacity(config.shuffle_buffer_entries.min(1024));

    loop {
        if context.stop.load(Ordering::Acquire) {
            break;
        }

        let file = match File::open(&path) {
            Ok(file) => file,
            Err(error) => {
                report_error(
                    &context,
                    format!("failed to open {}: {error}", path.display()),
                );
                break;
            }
        };

        let mut reader = match CompressedTrainingDataEntryReader::new(file) {
            Ok(reader) => reader,
            Err(CompressedReaderError::EndOfFile) => {
                if config.cyclic {
                    continue;
                }
                break;
            }
            Err(error) => {
                report_error(
                    &context,
                    format!(
                        "failed to create binpack reader for {}: {error}",
                        path.display()
                    ),
                );
                break;
            }
        };

        loop {
            if config.shuffle_chunks {
                if !push_shuffled_entries(
                    &mut reader,
                    &mut shuffled_entries,
                    &mut skip_decider,
                    &mut rng,
                    &context,
                ) {
                    break;
                }
            } else if !push_ordered_entries(
                &mut reader,
                &mut ordered_entries,
                &mut skip_decider,
                &mut rng,
                &context,
            ) {
                break;
            }
        }

        if !config.cyclic {
            break;
        }
    }

    if active_decoders.fetch_sub(1, Ordering::AcqRel) == 1 {
        context.position_queue.close();
    }
}

fn push_shuffled_entries(
    reader: &mut CompressedTrainingDataEntryReader<File>,
    buffer: &mut Vec<TrainingDataEntry>,
    skip_decider: &mut SkipDecider,
    rng: &mut SmallRng,
    context: &ThreadContext,
) -> bool {
    fill_shuffle_buffer(reader, buffer, context);
    while let Some(entry) = pop_shuffled_entry(buffer, rng) {
        if !emit_or_skip_decoded_entry(entry, context, skip_decider, rng) {
            return false;
        }

        if reader.has_next() {
            buffer.push(reader.next());
        }
    }

    reader.has_next() && !context.stop.load(Ordering::Acquire)
}

fn fill_shuffle_buffer(
    reader: &mut CompressedTrainingDataEntryReader<File>,
    buffer: &mut Vec<TrainingDataEntry>,
    context: &ThreadContext,
) {
    while buffer.len() < buffer.capacity()
        && reader.has_next()
        && !context.stop.load(Ordering::Acquire)
    {
        buffer.push(reader.next());
    }
}

fn pop_shuffled_entry(
    buffer: &mut Vec<TrainingDataEntry>,
    rng: &mut SmallRng,
) -> Option<TrainingDataEntry> {
    if buffer.is_empty() {
        return None;
    }

    let index = rng.gen_range(0..buffer.len());
    Some(buffer.swap_remove(index))
}

fn push_ordered_entries(
    reader: &mut CompressedTrainingDataEntryReader<File>,
    buffer: &mut VecDeque<TrainingDataEntry>,
    skip_decider: &mut SkipDecider,
    rng: &mut SmallRng,
    context: &ThreadContext,
) -> bool {
    while let Some(entry) = buffer.pop_front() {
        if !emit_or_skip_decoded_entry(entry, context, skip_decider, rng) {
            return false;
        }
    }

    if !reader.has_next() || context.stop.load(Ordering::Acquire) {
        return false;
    }

    buffer.push_back(reader.next());

    true
}

fn emit_or_skip_decoded_entry(
    entry: TrainingDataEntry,
    context: &ThreadContext,
    skip_decider: &mut SkipDecider,
    rng: &mut SmallRng,
) -> bool {
    context.stats.decoded_entries.fetch_add(1, Ordering::AcqRel);
    if skip_decider.should_skip(&entry, rng) {
        context.stats.skipped_entries.fetch_add(1, Ordering::AcqRel);
        return true;
    }

    context
        .position_queue
        .push_blocking(entry, &context.stop)
        .is_ok()
}

fn encoder_worker(
    _worker_index: usize,
    config: PipelineConfig,
    context: ThreadContext,
    active_encoders: Arc<AtomicUsize>,
) {
    let mut current_chunk: Option<HostBatchSlab> = None;

    while let Some(entry) = context.position_queue.pop_blocking(&context.stop) {
        if current_chunk.is_none() {
            current_chunk = context.free_chunks.pop_blocking(&context.stop);
            if current_chunk.is_none() {
                break;
            }
        }

        let chunk = current_chunk.as_mut().unwrap();
        chunk.push_entry(&entry, &config.feature_set);
        context.stats.encoded_entries.fetch_add(1, Ordering::AcqRel);

        if chunk.is_full() {
            let full_chunk = current_chunk.take().unwrap();
            if context
                .ready_chunks
                .push_blocking(full_chunk, &context.stop)
                .is_err()
            {
                break;
            }
        }
    }

    if let Some(chunk) = current_chunk.take() {
        if chunk.is_empty() {
            let _ = context.free_chunks.push_blocking(chunk, &context.stop);
        } else {
            let _ = context.ready_chunks.push_blocking(chunk, &context.stop);
        }
    }

    if active_encoders.fetch_sub(1, Ordering::AcqRel) == 1 {
        context.ready_chunks.close();
    }
}

fn assembler_worker(_config: PipelineConfig, context: ThreadContext) {
    let mut current_slab: Option<HostBatchSlab> = None;

    while let Some(mut chunk) = context.ready_chunks.pop_blocking(&context.stop) {
        let mut chunk_row = 0;

        while chunk_row < chunk.len() {
            if current_slab.is_none() {
                current_slab = context.free_slabs.pop_blocking(&context.stop);
                if current_slab.is_none() {
                    context.ready_slabs.close();
                    return;
                }
            }

            let slab = current_slab.as_mut().unwrap();
            chunk_row += slab.append_rows_from(&chunk, chunk_row);

            if slab.is_full() {
                let full_slab = current_slab.take().unwrap();
                if publish_ready_slab(full_slab, &context).is_err() {
                    context.ready_slabs.close();
                    return;
                }
            }
        }

        chunk.reset();
        if context
            .free_chunks
            .push_blocking(chunk, &context.stop)
            .is_err()
        {
            context.ready_slabs.close();
            return;
        }
    }

    if let Some(slab) = current_slab.take() {
        if slab.is_empty() {
            let _ = context.free_slabs.push_blocking(slab, &context.stop);
        } else {
            let _ = publish_ready_slab(slab, &context);
        }
    }

    context.ready_slabs.close();
}

fn publish_ready_slab(slab: HostBatchSlab, context: &ThreadContext) -> Result<(), HostBatchSlab> {
    context.ready_slabs.push_blocking(slab, &context.stop)?;
    context
        .stats
        .produced_batches
        .fetch_add(1, Ordering::AcqRel);
    Ok(())
}

fn report_error(context: &ThreadContext, message: String) {
    let mut guard = context.error.lock().unwrap();
    if guard.is_none() {
        *guard = Some(PipelineError::new(message));
    }
    drop(guard);

    context.stop.store(true, Ordering::Release);
    context.position_queue.close();
    context.ready_slabs.close();
    context.free_slabs.close();
}

fn desired_piece_count_weight(config: SkipConfig, piece_count: i32) -> f64 {
    let x = piece_count as f64;
    let x1 = 0.0;
    let y1 = config.pc_y1;
    let x2 = 16.0;
    let y2 = config.pc_y2;
    let x3 = 32.0;
    let y3 = config.pc_y3;
    let l1 = (x - x2) * (x - x3) / ((x1 - x2) * (x1 - x3));
    let l2 = (x - x1) * (x - x3) / ((x2 - x1) * (x2 - x3));
    let l3 = (x - x1) * (x - x2) / ((x3 - x1) * (x3 - x2));

    l1 * y1 + l2 * y2 + l3 * y3
}

fn is_capturing_move(entry: &TrainingDataEntry) -> bool {
    let to = entry.mv.to();
    let from = entry.mv.from();

    if to == sfbinpack::chess::coords::Square::NONE
        || from == sfbinpack::chess::coords::Square::NONE
    {
        return false;
    }

    let captured = entry.pos.piece_at(to);
    let moving = entry.pos.piece_at(from);

    captured != Piece::none() && captured.color() != moving.color()
}

fn is_in_check(entry: &TrainingDataEntry) -> bool {
    entry.pos.is_checked(entry.pos.side_to_move())
}

fn simple_eval(pos: &sfbinpack::chess::position::Position) -> i32 {
    let side_to_move_sign = if pos.side_to_move() == Color::White {
        1
    } else {
        -1
    };
    side_to_move_sign
        * (208 * material_count(pos, Color::White, PieceType::Pawn)
            - 208 * material_count(pos, Color::Black, PieceType::Pawn)
            + 781 * material_count(pos, Color::White, PieceType::Knight)
            - 781 * material_count(pos, Color::Black, PieceType::Knight)
            + 825 * material_count(pos, Color::White, PieceType::Bishop)
            - 825 * material_count(pos, Color::Black, PieceType::Bishop)
            + 1276 * material_count(pos, Color::White, PieceType::Rook)
            - 1276 * material_count(pos, Color::Black, PieceType::Rook)
            + 2538 * material_count(pos, Color::White, PieceType::Queen)
            - 2538 * material_count(pos, Color::Black, PieceType::Queen))
}

fn material_count(
    pos: &sfbinpack::chess::position::Position,
    color: Color,
    piece_type: PieceType,
) -> i32 {
    pos.pieces_bb_color(color, piece_type).count() as i32
}

fn score_result_prob(entry: &TrainingDataEntry) -> f64 {
    let m = entry.ply.min(240) as f64 / 64.0;
    let as_ = [-3.68389304, 30.07065921, -60.52878723, 149.53378557];
    let bs = [-2.0181857, 15.85685038, -29.83452023, 47.59078827];
    let a = (((as_[0] * m + as_[1]) * m + as_[2]) * m) + as_[3];
    let mut b = (((bs[0] * m + bs[1]) * m + bs[2]) * m) + bs[3];
    b *= 1.5;

    let x = ((100.0 * entry.score as f64) / 208.0).clamp(-2000.0, 2000.0);
    let w = 1.0 / (1.0 + ((a - x) / b).exp());
    let l = 1.0 / (1.0 + ((a + x) / b).exp());
    let d = 1.0 - w - l;

    if entry.result > 0 {
        w
    } else if entry.result < 0 {
        l
    } else {
        d
    }
}

fn feature_values_from_indices(features: &[i32]) -> Vec<f32> {
    features
        .iter()
        .map(|&feature| if feature == -1 { 0.0 } else { 1.0 })
        .collect()
}

fn random_seed() -> u64 {
    rand::thread_rng().gen()
}

pub(crate) fn default_encoder_chunk_size(batch_size: usize, encoding_threads: usize) -> usize {
    if encoding_threads <= 1 {
        return batch_size;
    }

    let target = batch_size
        .saturating_div(encoding_threads.saturating_mul(4).max(1))
        .max(1);

    target.clamp(512, 4096).min(batch_size)
}

pub(crate) fn default_chunk_queue_capacity(encoding_threads: usize) -> usize {
    encoding_threads.saturating_mul(2).max(4)
}

pub(crate) fn default_batch_slab_count(encoding_threads: usize) -> usize {
    encoding_threads.saturating_add(2).clamp(4, 8)
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use sfbinpack::chess::{
        coords::Square,
        piece::Piece,
        position::Position,
        r#move::{Move, MoveType},
    };
    use sfbinpack::CompressedTrainingDataEntryWriter;
    use tempfile::NamedTempFile;

    use super::*;

    #[test]
    fn skip_decider_matches_basic_cpp_conditions() {
        let mut decider = SkipDecider::new(SkipConfig {
            filtered: true,
            random_fen_skipping: 0,
            wld_filtered: false,
            early_fen_skipping: 4,
            simple_eval_skipping: 100,
            ..SkipConfig::default()
        });
        let mut rng = SmallRng::seed_from_u64(7);

        let early = make_entry(
            "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
            Move::normal(sq("e1"), sq("e2")),
            0,
            1,
            0,
        );
        assert!(decider.should_skip(&early, &mut rng));

        let capture = make_entry(
            "4k3/8/8/8/8/8/4p3/4K3 w - - 0 1",
            Move::normal(sq("e1"), sq("e2")),
            50,
            10,
            0,
        );
        assert!(decider.should_skip(&capture, &mut rng));

        let simple_eval_low = make_entry(
            "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
            Move::normal(sq("e1"), sq("e2")),
            50,
            10,
            0,
        );
        assert!(decider.should_skip(&simple_eval_low, &mut rng));
    }

    #[test]
    fn threaded_pipeline_matches_serial_rows_without_shuffle() {
        let feature_set =
            FeatureSet::from_str("Full_Threats+HalfKAv2_hm").expect("feature set should parse");
        let entries = sample_entries();
        let file = write_entries(&entries);
        let mut config =
            PipelineConfig::new(vec![file.path().to_path_buf()], feature_set.clone(), 2);
        config.encoding_threads = 1;
        config.slab_count = 2;
        config.shuffle_chunks = false;
        config.position_queue_capacity = 64;
        config.position_queue_high_watermark = 63;
        config.position_queue_low_watermark = 32;
        config.shuffle_buffer_entries = 8;
        config.seed = Some(11);

        let pipeline = BatchPipeline::new(config).expect("pipeline should start");
        let actual = collect_pipeline_rows(&pipeline).expect("pipeline should complete");
        let expected = entries
            .iter()
            .map(|entry| build_sparse_row_for_test(entry, &feature_set))
            .collect::<Vec<_>>();

        assert_eq!(actual, expected);
    }

    #[test]
    fn multithreaded_pipeline_matches_serial_multiset_with_shuffle() {
        let feature_set = FeatureSet::from_str("HalfKAv2_hm").expect("feature set should parse");
        let entries = sample_entries();
        let file_a = write_entries(&entries[..2]);
        let file_b = write_entries(&entries[2..]);
        let mut config = PipelineConfig::new(
            vec![file_a.path().to_path_buf(), file_b.path().to_path_buf()],
            feature_set.clone(),
            2,
        );
        config.encoding_threads = 2;
        config.slab_count = 3;
        config.shuffle_chunks = true;
        config.position_queue_capacity = 64;
        config.position_queue_high_watermark = 63;
        config.position_queue_low_watermark = 32;
        config.shuffle_buffer_entries = 2;
        config.seed = Some(19);

        let pipeline = BatchPipeline::new(config).expect("pipeline should start");
        let mut actual = collect_pipeline_rows(&pipeline).expect("pipeline should complete");
        let mut expected = entries
            .iter()
            .map(|entry| build_sparse_row_for_test(entry, &feature_set))
            .collect::<Vec<_>>();

        actual.sort_by_key(row_key);
        expected.sort_by_key(row_key);
        assert_eq!(actual, expected);
    }

    fn collect_pipeline_rows(pipeline: &BatchPipeline) -> Result<Vec<SparseRow>, PipelineError> {
        let mut rows = Vec::new();

        while let Some(batch) = pipeline.next_batch()? {
            for row in 0..batch.len() {
                rows.push(batch.copy_row(row));
            }
        }

        Ok(rows)
    }

    fn build_sparse_row_for_test(entry: &TrainingDataEntry, feature_set: &FeatureSet) -> SparseRow {
        crate::feature_extraction::build_sparse_row_for_feature_set(
            &entry.pos,
            entry.score as i32,
            entry.result as i32,
            feature_set,
        )
    }

    fn row_key(row: &SparseRow) -> String {
        format!("{row:?}")
    }

    fn sample_entries() -> Vec<TrainingDataEntry> {
        vec![
            make_entry(
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                Move::normal(sq("e2"), sq("e4")),
                12,
                1,
                0,
            ),
            make_entry(
                "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
                Move::castle(Square::E1, Square::H1),
                -44,
                8,
                1,
            ),
            make_entry(
                "4k3/1q6/8/3P4/8/8/6Q1/4K3 w - - 0 1",
                Move::normal(sq("g2"), sq("g7")),
                87,
                17,
                -1,
            ),
            make_entry(
                "rnbq1bnr/ppppkppp/8/4p3/3P4/2N5/PPP1PPPP/R1BQKBNR b KQ - 3 4",
                Move::normal(sq("e7"), sq("e6")),
                -120,
                9,
                0,
            ),
        ]
    }

    fn write_entries(entries: &[TrainingDataEntry]) -> NamedTempFile {
        let file = NamedTempFile::new().expect("temporary binpack file should be created");
        let writer_file = file.reopen().expect("temporary file should be reopenable");
        let mut writer = CompressedTrainingDataEntryWriter::new(writer_file)
            .expect("binpack writer should be created");

        for entry in entries {
            writer
                .write_entry(entry)
                .expect("sample entry should be written");
        }
        writer.flush_and_end();

        file
    }

    fn make_entry(fen: &str, mv: Move, score: i16, ply: u16, result: i16) -> TrainingDataEntry {
        TrainingDataEntry {
            pos: Position::from_fen(fen).expect("FEN should parse"),
            mv,
            score,
            ply,
            result,
        }
    }

    fn sq(name: &str) -> Square {
        Square::from_string(name).expect("square should parse")
    }

    #[allow(dead_code)]
    fn _promotion_entry() -> TrainingDataEntry {
        make_entry(
            "4k3/4P3/8/8/8/8/8/4K3 w - - 0 1",
            Move::new(
                sq("e7"),
                Square::E8,
                MoveType::Promotion,
                Piece::WHITE_QUEEN,
            ),
            300,
            30,
            1,
        )
    }
}
