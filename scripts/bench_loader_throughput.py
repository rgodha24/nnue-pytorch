#!/usr/bin/env python3
"""
Loader throughput benchmark for the exact Rust training data path used by the
long H100 run, except for dataset locations.

This keeps the same training-side shape:
- Rust BatchStream
- tensor wrapping matching RustSparseBatchProvider
- FixedNumBatchesDataset
- DataLoader(batch_size=None, num_workers=0)

It also exposes explicit total/decode/encode thread overrides so skip-heavy
loader sweeps can test alternative splits and oversubscription.
"""

from __future__ import annotations

import os
import queue
import sys
import time
from dataclasses import dataclass, field

# Repo root on sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import tyro
from torch.utils.data import DataLoader, IterableDataset

import data_loader
from data_loader.config import DataloaderSkipConfig
from model.config import NNUELightningConfig
from model.model import NNUEModel
from model.quantize import QuantizationConfig


DEFAULT_DATASETS: tuple[str, ...] = (
    "/tmp/nnue-data/T60T70wIsRightFarseerT60T74T75T76.split_0.binpack",
    "/tmp/nnue-data/T60T70wIsRightFarseerT60T74T75T76.split_1.binpack",
    "/tmp/nnue-data/T60T70wIsRightFarseerT60T74T75T76.split_2.binpack",
    "/tmp/nnue-data/T60T70wIsRightFarseerT60T74T75T76.split_3.binpack",
    "/tmp/nnue-data/T60T70wIsRightFarseerT60T74T75T76.split_4.binpack",
    "/tmp/nnue-data/dfrc_n5000.binpack",
    "/tmp/nnue-data/multinet_pv-2_diff-100_nodes-5000.binpack",
    "/tmp/nnue-data/nodes5000pv2_UHO.binpack",
    "/tmp/nnue-data/wrongIsRight_nodes5000pv2.binpack",
)


@dataclass(kw_only=True)
class BenchConfig:
    datasets: tuple[str, ...] = DEFAULT_DATASETS
    """Dataset paths. Defaults to the /tmp copy of the training data."""

    features: str = "Full_Threats+HalfKAv2_hm^"
    """Matches the training command exactly."""

    batch_size: int = 65536
    """Matches the training command exactly."""

    num_workers: int = 1
    """Kept for parity with train.py. The Rust path uses internal threads."""

    loader_threads: int = -1
    """Train.py-compatible total-thread knob when total_threads is not set."""

    total_threads: int | None = None
    """Explicit total Rust loader threads. May exceed CPU affinity size."""

    decode_threads: int | None = None
    """Explicit Rust decoder thread count."""

    encode_threads: int | None = None
    """Explicit Rust encoder thread count."""

    slab_count: int | None = None
    """Override batch slab count. Default matches training heuristic."""

    shuffle_buffer_entries: int = 16384
    """Matches the Rust training path."""

    pin_memory: bool = True
    """Matches train.py default."""

    data_loader_queue_size: int = 16
    """Matches train.py default."""

    dataloader_config: DataloaderSkipConfig = field(
        default_factory=lambda: DataloaderSkipConfig(
            filtered=True,
            wld_filtered=True,
            random_fen_skipping=10,
            early_fen_skipping=12,
        )
    )
    """Matches the long-run training command for loader-relevant flags."""

    warmup_batches: int = 5
    timed_batches: int = 100
    first_batch_poll_interval: float = 5.0
    """How often to print progress while waiting for the first batch."""


def _input_feature_name(features: str) -> str:
    cfg = NNUELightningConfig(features=features)
    return NNUEModel(
        cfg.features,
        cfg.model_config,
        QuantizationConfig(),
    ).input_feature_name


def _default_total_threads(loader_threads: int) -> int:
    if loader_threads > 0:
        return loader_threads
    try:
        cpu_count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpu_count = os.cpu_count() or 8
    return max(1, cpu_count - 1)


def _skip_heavy(config: DataloaderSkipConfig) -> bool:
    return (
        config.filtered
        or config.wld_filtered
        or config.random_fen_skipping > 0
        or config.early_fen_skipping >= 0
        or (config.simple_eval_skipping is not None and config.simple_eval_skipping > 0)
    )


def _resolve_thread_config(args: BenchConfig) -> tuple[int, int, int, int]:
    explicit_total = args.total_threads
    if explicit_total is None:
        explicit_total = _default_total_threads(args.loader_threads)

    decode = args.decode_threads
    encode = args.encode_threads

    if decode is not None and decode <= 0:
        raise ValueError("decode_threads must be > 0")
    if encode is not None and encode <= 0:
        raise ValueError("encode_threads must be > 0")
    if explicit_total <= 0:
        raise ValueError("total_threads must be > 0")

    if decode is not None and encode is not None:
        if explicit_total != decode + encode:
            raise ValueError(
                f"total_threads={explicit_total} must equal decode_threads + encode_threads ({decode + encode})"
            )
    elif decode is not None:
        if decode >= explicit_total:
            raise ValueError("decode_threads must be < total_threads")
        encode = explicit_total - decode
    elif encode is not None:
        if encode >= explicit_total:
            raise ValueError("encode_threads must be < total_threads")
        decode = explicit_total - encode
    else:
        if _skip_heavy(args.dataloader_config):
            # See dataset.py: one encode thread per ~16 cores (ceil division).
            encode = max(1, (explicit_total + 15) // 16)
        else:
            encode = max(1, explicit_total // 4)
        decode = max(1, explicit_total - encode)
        explicit_total = decode + encode

    if args.slab_count is None:
        slab_count = max(8, encode + max(4, encode // 2))
    else:
        if args.slab_count <= 0:
            raise ValueError("slab_count must be > 0")
        slab_count = args.slab_count

    return explicit_total, decode, encode, slab_count


class BenchRustProvider:
    def __init__(self, args: BenchConfig, input_feature_name: str):
        import nnue_loader

        batch_stream_cls = getattr(nnue_loader, "BatchStream")
        total_threads, decode_threads, encode_threads, slab_count = (
            _resolve_thread_config(args)
        )

        self.resolved_total_threads = total_threads
        self.resolved_decode_threads = decode_threads
        self.resolved_encode_threads = encode_threads
        self.resolved_slab_count = slab_count
        self.resolved_shuffle_buffer_entries = args.shuffle_buffer_entries

        cfg = args.dataloader_config
        self._stream = batch_stream_cls(
            input_feature_name,
            list(args.datasets),
            args.batch_size,
            total_threads=total_threads,
            decode_threads=decode_threads,
            encode_threads=encode_threads,
            slab_count=slab_count,
            shuffle_buffer_entries=args.shuffle_buffer_entries,
            cyclic=True,
            filtered=cfg.filtered,
            random_fen_skipping=int(cfg.random_fen_skipping),
            wld_filtered=cfg.wld_filtered,
            early_fen_skipping=int(cfg.early_fen_skipping),
            simple_eval_skipping=int(cfg.simple_eval_skipping or 0),
            param_index=int(cfg.param_index),
            pc_y1=float(cfg.pc_y1),
            pc_y2=float(cfg.pc_y2),
            pc_y3=float(cfg.pc_y3),
            rank=0,
            world_size=1,
        )

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
                pass
            self._ones_buffer = buf
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


class BenchSparseBatchDataset(IterableDataset):
    def __init__(self, args: BenchConfig, input_feature_name: str):
        super().__init__()
        self.args = args
        self.input_feature_name = input_feature_name

    def __iter__(self):
        return BenchRustProvider(self.args, self.input_feature_name)


def make_loader(args: BenchConfig, input_feature_name: str) -> DataLoader:
    stream = BenchSparseBatchDataset(args, input_feature_name)
    dataset = data_loader.FixedNumBatchesDataset(
        stream,
        num_batches=args.warmup_batches + args.timed_batches,
        pin_memory=args.pin_memory,
        queue_size_limit=args.data_loader_queue_size,
    )
    return DataLoader(
        dataset,
        batch_size=None,
        batch_sampler=None,
        num_workers=0,
    )


def _format_stats(stats: dict) -> str:
    keys = [
        "decoded_entries",
        "skipped_entries",
        "encoded_entries",
        "produced_batches",
        "decoded_queue_len",
        "ready_queue_len",
        "free_queue_len",
    ]
    parts = []
    for key in keys:
        if key in stats:
            parts.append(f"{key}={stats[key]}")
    return " ".join(parts)


def _read_stream_stats(dataset) -> dict | None:
    provider = getattr(dataset, "iter", None)
    stream = getattr(provider, "_stream", None)
    if stream is None or not hasattr(stream, "stats"):
        return None
    try:
        stats = stream.stats()
    except Exception:
        return None
    if not isinstance(stats, dict):
        return None
    return stats


def _maybe_get_stream_stats(dataset) -> str | None:
    stats = _read_stream_stats(dataset)
    if stats is None:
        return None
    return _format_stats(stats)


def _stat_delta(end: dict | None, start: dict | None, key: str) -> int | None:
    if end is None or start is None or key not in end or key not in start:
        return None
    return int(end[key]) - int(start[key])


def _provider_summary(dataset) -> str | None:
    provider = getattr(dataset, "iter", None)
    if provider is None:
        return None
    keys = [
        "resolved_total_threads",
        "resolved_decode_threads",
        "resolved_encode_threads",
        "resolved_slab_count",
        "resolved_shuffle_buffer_entries",
    ]
    if not all(hasattr(provider, key) for key in keys):
        return None
    return (
        "resolved_loader="
        f"total_threads={provider.resolved_total_threads} "
        f"decode_threads={provider.resolved_decode_threads} "
        f"encode_threads={provider.resolved_encode_threads} "
        f"slab_count={provider.resolved_slab_count} "
        f"shuffle_buffer_entries={provider.resolved_shuffle_buffer_entries}"
    )


def wait_for_first_batch(dataset, poll_interval: float):
    if not hasattr(dataset, "_start_prefetching") or not hasattr(
        dataset, "_prefetch_queue"
    ):
        raise RuntimeError(
            "dataset does not expose prefetch internals for first-batch polling"
        )

    dataset._start_prefetching()
    provider_text = _provider_summary(dataset)
    if provider_text:
        print(provider_text, flush=True)

    started = time.perf_counter()
    while True:
        try:
            item = dataset._prefetch_queue.get(timeout=poll_interval)
        except queue.Empty:
            elapsed = time.perf_counter() - started
            stats_text = _maybe_get_stream_stats(dataset)
            if stats_text:
                print(
                    f"waiting_for_first_batch elapsed={elapsed:.1f}s {stats_text}",
                    flush=True,
                )
            else:
                print(f"waiting_for_first_batch elapsed={elapsed:.1f}s", flush=True)
            continue

        if item is None:
            raise StopIteration("End of dataset reached before first batch")
        if isinstance(item, Exception):
            raise item
        return item, time.perf_counter() - started


def consume_one(it) -> tuple:
    batch = next(it)
    _ = batch[0].shape[0]
    _ = batch[2].shape[1]
    return batch


def main() -> None:
    args = tyro.cli(BenchConfig)

    if args.warmup_batches < 0:
        raise SystemExit("warmup_batches must be >= 0")
    if args.timed_batches <= 0:
        raise SystemExit("timed_batches must be > 0")
    if not args.datasets:
        raise SystemExit("no datasets configured")

    for path in args.datasets:
        if not os.path.isfile(path):
            raise SystemExit(f"missing dataset: {path}")

    input_feature_name = _input_feature_name(args.features)
    total_threads, decode_threads, encode_threads, slab_count = _resolve_thread_config(
        args
    )

    try:
        cpu_affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpu_affinity = os.cpu_count() or 0

    print(
        f"features={args.features!r}\n"
        f"input_feature_name={input_feature_name!r}\n"
        f"cpu_affinity={cpu_affinity}\n"
        f"batch_size={args.batch_size}\n"
        f"num_workers={args.num_workers}\n"
        f"loader_threads={args.loader_threads}\n"
        f"total_threads={args.total_threads}\n"
        f"decode_threads={args.decode_threads}\n"
        f"encode_threads={args.encode_threads}\n"
        f"slab_count={args.slab_count}\n"
        f"resolved_total_threads={total_threads}\n"
        f"resolved_decode_threads={decode_threads}\n"
        f"resolved_encode_threads={encode_threads}\n"
        f"resolved_slab_count={slab_count}\n"
        f"shuffle_buffer_entries={args.shuffle_buffer_entries}\n"
        f"pin_memory={args.pin_memory}\n"
        f"data_loader_queue_size={args.data_loader_queue_size}\n"
        f"first_batch_poll_interval={args.first_batch_poll_interval}\n"
        f"DataloaderSkipConfig={args.dataloader_config}\n"
        f"files={list(args.datasets)}\n"
        f"warmup_batches={args.warmup_batches} timed_batches={args.timed_batches}\n"
    )

    print(
        "Train-path loader (Rust BatchStream -> FixedNumBatchesDataset -> DataLoader)...",
        flush=True,
    )

    loader = make_loader(args, input_feature_name)
    fixed_dataset = loader.dataset

    if args.warmup_batches > 0:
        _, first_batch_latency = wait_for_first_batch(
            fixed_dataset, args.first_batch_poll_interval
        )
        print(f"first_batch_latency={first_batch_latency:.3f}s", flush=True)

    it = iter(loader)
    remaining_warmup = max(0, args.warmup_batches - 1)
    for _ in range(remaining_warmup):
        consume_one(it)

    start_stats = _read_stream_stats(fixed_dataset)
    t0 = time.perf_counter()
    for _ in range(args.timed_batches):
        consume_one(it)
    elapsed = time.perf_counter() - t0
    end_stats = _read_stream_stats(fixed_dataset)

    batches_per_second = args.timed_batches / elapsed
    positions_per_second = (args.timed_batches * args.batch_size) / elapsed
    print(
        f"  {elapsed:.3f}s for {args.timed_batches} batches -> "
        f"{batches_per_second:.2f} batches/s ({positions_per_second:,.0f} positions/s)"
    )

    decoded = _stat_delta(end_stats, start_stats, "decoded_entries")
    skipped = _stat_delta(end_stats, start_stats, "skipped_entries")
    encoded = _stat_delta(end_stats, start_stats, "encoded_entries")
    produced = _stat_delta(end_stats, start_stats, "produced_batches")
    if decoded is not None:
        skipped_text = f"{skipped:,}" if skipped is not None else "n/a"
        encoded_text = f"{encoded:,}" if encoded is not None else "n/a"
        produced_text = f"{produced:,}" if produced is not None else "n/a"
        print(
            f"timed_stats decoded={decoded:,} skipped={skipped_text} encoded={encoded_text} produced_batches={produced_text}"
        )
        skipped_rate_text = (
            f"{skipped / elapsed:,.0f}/s" if skipped is not None else "n/a"
        )
        encoded_rate_text = (
            f"{encoded / elapsed:,.0f}/s" if encoded is not None else "n/a"
        )
        print(
            f"timed_rates decoded={decoded / elapsed:,.0f}/s skipped={skipped_rate_text} encoded={encoded_rate_text}"
        )
        if decoded > 0 and encoded is not None and skipped is not None:
            print(
                f"timed_ratios keep={(100.0 * encoded / decoded):.3f}% skip={(100.0 * skipped / decoded):.3f}%"
            )


if __name__ == "__main__":
    main()
