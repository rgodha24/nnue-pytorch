import os
import threading
import queue

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
        ddp_config: DataloaderDDPConfig = None,
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
                ddp_config,
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


class TrainingDataProvider:
    def __init__(
        self,
        feature_set: str,
        create_stream,
        destroy_stream,
        fetch_next,
        destroy_part,
        filenames: list[str],
        cyclic,
        num_workers,
        batch_size=None,
        config: DataloaderSkipConfig = DataloaderSkipConfig(),
        ddp_config: DataloaderDDPConfig = None,
    ):
        self.feature_set = feature_set.encode("utf-8")
        self.create_stream = create_stream
        self.destroy_stream = destroy_stream
        self.fetch_next = fetch_next
        self.destroy_part = destroy_part
        self.filenames = filenames
        self.cyclic = cyclic
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.config = config

        if batch_size:
            self.stream = self.create_stream(
                self.feature_set,
                self.num_workers,
                self.filenames,
                batch_size,
                cyclic,
                config,
                ddp_config,
            )
        else:
            self.stream = self.create_stream(
                self.feature_set,
                self.num_workers,
                self.filenames,
                cyclic,
                config,
                ddp_config,
            )

    def __iter__(self):
        return self

    def __next__(self):
        v = self.fetch_next(self.stream)

        if v:
            tensors = v.contents.get_tensors("cpu")
            self.destroy_part(v)
            return tensors
        else:
            raise StopIteration

    def __del__(self):
        self.destroy_stream(self.stream)


class CppSparseBatchProvider(TrainingDataProvider):
    def __init__(
        self,
        feature_set: str,
        filenames: list[str],
        batch_size,
        cyclic=True,
        num_workers=1,
        config: DataloaderSkipConfig = DataloaderSkipConfig(),
        ddp_config: DataloaderDDPConfig = None,
    ):
        super().__init__(
            feature_set,
            stream.create_sparse_batch_stream,
            stream.destroy_sparse_batch_stream,
            stream.fetch_next_sparse_batch,
            stream.destroy_sparse_batch,
            filenames,
            cyclic,
            num_workers,
            batch_size,
            config,
            ddp_config,
        )


class RustSparseBatchProvider:
    """Iterator that pulls batches from the Rust nnue_loader.BatchStream and
    converts them to torch tensors. The Rust loader replaces the C++
    SparseBatchProvider as the default training path."""

    def __init__(
        self,
        feature_set: str,
        filenames: list[str],
        batch_size,
        cyclic: bool = True,
        num_workers: int = 1,
        config: DataloaderSkipConfig = DataloaderSkipConfig(),
        ddp_config: DataloaderDDPConfig = None,
    ):
        import nnue_loader

        rank = 0
        world_size = 1
        if ddp_config is not None:
            rank = int(ddp_config.rank)
            world_size = int(ddp_config.world_size)

        # Tuned defaults derived from sweeping the AMD Ryzen 9 9950X (16
        # physical cores / 32 threads). Decoder threads are the bottleneck
        # for skip-heavy workloads (filtering rejects ~99% of positions on
        # the default UHO datasets), but feature extraction is also
        # expensive enough that we want enough encoders to fully drain the
        # decoded queue. The Rust loader will further refine the split based
        # on `total_threads` if you don't specify both.
        #
        # On slurm/cgroup-restricted hosts, `os.cpu_count()` returns the
        # host's full CPU count; `sched_getaffinity` returns the cgroup
        # allocation, which is what we actually have to spend.
        try:
            cpu_count = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            cpu_count = os.cpu_count() or 8
        # Reserve one thread for the consumer / pin-memory worker.
        total_threads = max(1, cpu_count - 1)

        # `num_workers` is the legacy C++ knob; if the user pinned it to
        # something explicit we honour that as a hint, but the Rust loader
        # uses dedicated decode/encode pools instead of one shared worker
        # pool, so we don't map it 1:1.
        if num_workers and num_workers > 1:
            total_threads = max(total_threads, num_workers)

        skip_heavy = (
            config.filtered
            or config.wld_filtered
            or config.random_fen_skipping > 0
            or config.early_fen_skipping >= 0
            or (
                config.simple_eval_skipping is not None
                and config.simple_eval_skipping > 0
            )
        )

        if skip_heavy:
            # ~60/40 decode/encode split. Decoders process ~10x more raw
            # entries than encoders end up keeping, but feature extraction
            # for Full_Threats+HalfKAv2_hm is expensive enough that the
            # encoders also need significant cores.
            decode_threads = max(1, (total_threads * 5) // 8)
        else:
            # Without skipping, encoders dominate.
            decode_threads = max(1, total_threads // 4)
        encode_threads = max(1, total_threads - decode_threads)

        # Slab pool: at least one slab per encoder + a few buffered for the
        # consumer. Bigger pools waste memory without throughput gains
        # beyond ~1.5x encoders.
        slab_count = max(8, encode_threads + max(4, encode_threads // 2))

        self._stream = nnue_loader.BatchStream(
            feature_set.replace("^", ""),
            list(filenames),
            batch_size,
            total_threads=total_threads,
            decode_threads=decode_threads,
            encode_threads=encode_threads,
            slab_count=slab_count,
            shuffle_buffer_entries=16384,
            cyclic=cyclic,
            filtered=config.filtered,
            random_fen_skipping=int(config.random_fen_skipping),
            wld_filtered=config.wld_filtered,
            early_fen_skipping=int(config.early_fen_skipping),
            simple_eval_skipping=int(config.simple_eval_skipping or 0),
            param_index=int(config.param_index),
            pc_y1=float(config.pc_y1),
            pc_y2=float(config.pc_y2),
            pc_y3=float(config.pc_y3),
            rank=rank,
            world_size=world_size,
        )

        # Reused all-ones buffer for feature_values. The CUDA kernel breaks on
        # index == -1, so values at padded slots are never read; active slots
        # always have value 1.0. A single shared buffer therefore works for
        # every batch and eliminates per-batch allocation/pin cost.
        self._ones_buffer: torch.Tensor | None = None

    def __iter__(self):
        return self

    def __next__(self):
        batch = self._stream.next_batch()
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

        rows, cols = white_indices.shape
        if (
            self._ones_buffer is None
            or self._ones_buffer.shape[1] != cols
            or self._ones_buffer.shape[0] < rows
        ):
            buf = torch.ones((rows, cols), dtype=torch.float32)
            try:
                buf = buf.pin_memory()
            except RuntimeError:
                # No CUDA available — fall back to a regular tensor.
                pass
            self._ones_buffer = buf
        # Slicing the first dim of a contiguous tensor stays contiguous and
        # preserves the pinned-memory flag, so the prefetch worker's
        # pin_memory() call is a no-op for this tensor.
        values = self._ones_buffer[:rows]

        return (
            us,
            them,
            white_indices,
            values,
            black_indices,
            values,
            outcome,
            score,
            psqt_indices,
            layer_stack_indices,
        )

    def __del__(self):
        if hasattr(self, "_stream"):
            self._stream.close()


# The Rust loader is now the default training data provider.
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
        ddp_config: DataloaderDDPConfig = None,
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
        self.iter = None  # Deferred to _start_prefetching
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
