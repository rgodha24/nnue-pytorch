import queue
import threading
from collections.abc import Iterator
from typing import cast

import nnue_loader
import torch
from torch.utils.data import Dataset

from . import stream
from .config import DataloaderSkipConfig, DataloaderDDPConfig


def _recursive_pin(obj):
    if isinstance(obj, torch.Tensor):
        return obj.pin_memory()
    elif isinstance(obj, dict):
        return {k: _recursive_pin(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(_recursive_pin(v) for v in obj)
    return obj


class FenBatchProvider:
    def __init__(
        self,
        filename,
        cyclic,
        num_workers,
        batch_size=None,
        config: DataloaderSkipConfig = DataloaderSkipConfig(),
        ddp_config: DataloaderDDPConfig | None = None,
    ):
        self.filename = filename
        self.cyclic = cyclic
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.config = config

        if batch_size:
            self.stream = stream.create_fen_batch_stream(
                self.num_workers,
                [self.filename],
                batch_size,
                cyclic,
                config,
                cast(DataloaderDDPConfig, ddp_config),
            )
        else:
            # doesnt work yet
            assert False
            # self.stream = make_fen_batch_stream(
            #     self.num_workers,
            #     [self.filename],
            #     cyclic,
            #     config=config
            # )

    def __iter__(self):
        return self

    def __next__(self):
        v = stream.fetch_next_fen_batch(self.stream)

        if v:
            fens = v.contents.get_fens()
            stream.destroy_fen_batch(v)
            return fens
        else:
            raise StopIteration

    def __del__(self):
        stream.destroy_fen_batch_stream(self.stream)


class RustSparseBatchProvider:
    def __init__(
        self,
        feature_set: str,
        filenames: list[str],
        batch_size,
        cyclic=True,
        num_workers=1,
        config: DataloaderSkipConfig = DataloaderSkipConfig(),
        ddp_config: DataloaderDDPConfig | None = None,
    ):
        if ddp_config is not None and ddp_config.world_size != 1:
            raise NotImplementedError(
                "Rust dataloader DDP support is not implemented yet"
            )

        decoder_threads = max(1, len(filenames))
        skip_heavy = (
            config.filtered
            or config.wld_filtered
            or config.random_fen_skipping > 0
            or config.early_fen_skipping >= 0
            or config.simple_eval_skipping > 0
        )

        if skip_heavy:
            encoding_threads = max(1, min(num_workers, decoder_threads * 2))
        else:
            encoding_threads = max(1, num_workers)

        slab_count = max(4, min(encoding_threads + 2, 8))
        position_queue_capacity = max(1 << 16, batch_size * max(4, decoder_threads * 2))
        position_queue_high_watermark = position_queue_capacity * 7 // 8
        position_queue_low_watermark = position_queue_capacity // 2
        shuffle_buffer_entries = max(65_536, min(batch_size * 2, 262_144))

        self.stream = nnue_loader.BatchStream(
            feature_set.replace("^", ""),
            filenames,
            batch_size,
            encoding_threads=encoding_threads,
            slab_count=slab_count,
            position_queue_capacity=position_queue_capacity,
            position_queue_high_watermark=position_queue_high_watermark,
            position_queue_low_watermark=position_queue_low_watermark,
            shuffle_buffer_entries=shuffle_buffer_entries,
            cyclic=cyclic,
            filtered=config.filtered,
            random_fen_skipping=config.random_fen_skipping,
            wld_filtered=config.wld_filtered,
            early_fen_skipping=config.early_fen_skipping,
            simple_eval_skipping=config.simple_eval_skipping,
            param_index=config.param_index,
            pc_y1=config.pc_y1,
            pc_y2=config.pc_y2,
            pc_y3=config.pc_y3,
        )

    def __iter__(self):
        return self

    def __next__(self):
        batch = self.stream.next_batch()
        if batch is None:
            raise StopIteration

        us = torch.from_numpy(batch["is_white"])
        them = 1.0 - us
        white_indices = torch.from_numpy(batch["white"])
        black_indices = torch.from_numpy(batch["black"])
        outcome = torch.from_numpy(batch["outcome"])
        score = torch.from_numpy(batch["score"])
        psqt_indices = torch.from_numpy(batch["psqt_indices"])
        layer_stack_indices = torch.from_numpy(batch["layer_stack_indices"])

        return (
            us,
            them,
            white_indices,
            black_indices,
            outcome,
            score,
            psqt_indices,
            layer_stack_indices,
        )

    def __del__(self):
        if hasattr(self, "stream"):
            self.stream.close()


SparseBatchProvider = RustSparseBatchProvider


class SparseBatchDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        feature_set: str,
        filenames: list[str],
        batch_size,
        cyclic=True,
        num_workers=1,
        config: DataloaderSkipConfig = DataloaderSkipConfig(),
        ddp_config: DataloaderDDPConfig | None = None,
    ):
        super().__init__()
        self.feature_set = feature_set
        self.filenames = filenames
        self.batch_size = batch_size
        self.cyclic = cyclic
        self.num_workers = num_workers
        self.config = config
        self.ddp_config = ddp_config

    def __iter__(self):
        return SparseBatchProvider(
            self.feature_set,
            self.filenames,
            self.batch_size,
            cyclic=self.cyclic,
            num_workers=self.num_workers,
            config=self.config,
            ddp_config=self.ddp_config,
        )


class FixedNumBatchesDataset(Dataset):
    def __init__(self, dataset, num_batches, pin_memory=False, queue_size_limit=None):
        super().__init__()
        self.dataset = dataset
        self.iter: Iterator | None = None  # Deferred to _start_prefetching
        self.num_batches = num_batches
        self.pin_memory = pin_memory
        if queue_size_limit is None:
            queue_size_limit = 10 if pin_memory else 100

        self._prefetch_queue = queue.Queue(maxsize=queue_size_limit)
        self._prefetch_thread = None
        self._stop_prefetching = threading.Event()
        self._prefetch_started = False
        self._lock = threading.Lock()

    def _safe_put(self, item):
        """Helper to ensure we don't hang on shutdown if queue is full."""
        while not self._stop_prefetching.is_set():
            try:
                self._prefetch_queue.put(item, timeout=1.0)
                break
            except queue.Full:
                continue

    def _prefetch_worker(self):
        try:
            while not self._stop_prefetching.is_set():
                try:
                    assert self.iter is not None
                    item = next(self.iter)
                    # Pin memory on worker thread if enabled.
                    if self.pin_memory:
                        item = _recursive_pin(item)
                    self._safe_put(item)
                except StopIteration:
                    self._safe_put(None)
                    break
        except Exception as e:
            self._safe_put(e)

    def _start_prefetching(self):
        with self._lock:
            if not self._prefetch_started:
                self.iter = iter(self.dataset)
                self._prefetch_thread = threading.Thread(
                    target=self._prefetch_worker, daemon=True
                )
                self._prefetch_thread.start()
                self._prefetch_started = True

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        self._start_prefetching()

        try:
            assert self.iter is not None
            item = self._prefetch_queue.get(timeout=300.0)  # 300 second timeout

            if item is None:
                raise StopIteration("End of dataset reached")
            elif isinstance(item, Exception):
                raise item

            return item

        except queue.Empty:
            raise RuntimeError("Prefetch timeout - no data available")

    def __del__(self):
        if hasattr(self, "_stop_prefetching"):
            self._stop_prefetching.set()
        if hasattr(self, "_prefetch_thread") and self._prefetch_thread:
            self._prefetch_thread.join(timeout=1.0)
