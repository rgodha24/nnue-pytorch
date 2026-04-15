#!/usr/bin/env python3

from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import tyro


ROOT = Path(__file__).resolve().parent.parent
BENCH_SCRIPT = ROOT / "scripts" / "bench_loader_throughput.py"


@dataclass(kw_only=True)
class SweepConfig:
    total_threads: tuple[int, ...] = ()
    """Explicit totals to test. Empty = a small default ladder."""

    min_total_threads: int | None = None
    """Lower bound for custom total-thread ranges."""

    max_total_threads: int | None = None
    """Upper bound for custom total-thread ranges."""

    total_step: int | None = None
    """Step for custom total-thread ranges. Default: one affinity-width."""

    decode_percents: tuple[int, ...] = (75, 88, 94)
    """Small default split set. On 16 threads this gives 12/4, 14/2, 15/1."""

    batch_size: int = 65536
    features: str = "Full_Threats+HalfKAv2_hm^"
    num_workers: int = 1
    shuffle_buffer_entries: int = 16384
    warmup_batches: int = 5
    timed_batches: int = 100
    first_batch_poll_interval: float = 5.0
    timeout_seconds: float = 1800.0
    output_csv: str | None = None
    show_full_output: bool = False
    stop_on_error: bool = False


@dataclass
class SweepResult:
    total_threads: int
    decode_threads: int
    encode_threads: int
    returncode: int
    cpu_affinity: int | None = None
    first_batch_latency_s: float | None = None
    elapsed_s: float | None = None
    timed_batches: int | None = None
    batches_per_s: float | None = None
    positions_per_s: int | None = None
    decoded_per_s: float | None = None
    skipped_per_s: float | None = None
    encoded_per_s: float | None = None
    keep_ratio_pct: float | None = None
    skip_ratio_pct: float | None = None
    stdout: str = ""
    stderr: str = ""


def _coerce_text(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def _format_float(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _format_int(value: int | None) -> str:
    if value is None:
        return "-"
    return f"{value:,}"


def _parse_int(text: str) -> int:
    return int(text.replace(",", ""))


def _parse_rate(text: str) -> float | None:
    text = text.strip()
    if text == "n/a":
        return None
    if text.endswith("/s"):
        text = text[:-2]
    return float(text.replace(",", ""))


def _auto_totals(cfg: SweepConfig, affinity: int) -> tuple[int, ...]:
    if cfg.total_threads:
        return tuple(sorted({value for value in cfg.total_threads if value > 1}))

    if cfg.min_total_threads is None and cfg.max_total_threads is None:
        return tuple(sorted({affinity, 2 * affinity, 3 * affinity}))

    min_total = cfg.min_total_threads or affinity
    max_total = cfg.max_total_threads or min_total
    if min_total < 2:
        min_total = 2
    if max_total < min_total:
        max_total = min_total
    if cfg.total_step is None:
        step = max(1, affinity)
    else:
        step = cfg.total_step
    if step <= 0:
        raise ValueError("total_step must be > 0")

    totals = list(range(min_total, max_total + 1, step))
    if totals[-1] != max_total:
        totals.append(max_total)
    return tuple(sorted(set(totals)))


def _cases(cfg: SweepConfig, affinity: int) -> list[tuple[int, int, int]]:
    totals = _auto_totals(cfg, affinity)
    cases: set[tuple[int, int, int]] = set()
    for total in totals:
        for pct in cfg.decode_percents:
            if not 1 <= pct <= 99:
                raise ValueError("decode_percents entries must be in [1, 99]")
            decode = max(1, min(total - 1, round(total * pct / 100.0)))
            encode = total - decode
            if encode <= 0:
                continue
            cases.add((total, decode, encode))
    return sorted(cases)


def _build_command(
    cfg: SweepConfig, total_threads: int, decode_threads: int, encode_threads: int
) -> list[str]:
    return [
        sys.executable,
        str(BENCH_SCRIPT),
        f"--features={cfg.features}",
        f"--batch-size={cfg.batch_size}",
        f"--num-workers={cfg.num_workers}",
        f"--total-threads={total_threads}",
        f"--decode-threads={decode_threads}",
        f"--encode-threads={encode_threads}",
        f"--shuffle-buffer-entries={cfg.shuffle_buffer_entries}",
        f"--warmup-batches={cfg.warmup_batches}",
        f"--timed-batches={cfg.timed_batches}",
        f"--first-batch-poll-interval={cfg.first_batch_poll_interval}",
    ]


def _parse_output(
    total_threads: int,
    decode_threads: int,
    encode_threads: int,
    proc: subprocess.CompletedProcess[str],
) -> SweepResult:
    stdout = proc.stdout
    result = SweepResult(
        total_threads=total_threads,
        decode_threads=decode_threads,
        encode_threads=encode_threads,
        returncode=proc.returncode,
        stdout=stdout,
        stderr=proc.stderr,
    )

    cpu_match = re.search(r"^cpu_affinity=(\d+)$", stdout, re.MULTILINE)
    if cpu_match:
        result.cpu_affinity = int(cpu_match.group(1))

    first_match = re.search(r"^first_batch_latency=([0-9.]+)s$", stdout, re.MULTILINE)
    if first_match:
        result.first_batch_latency_s = float(first_match.group(1))

    throughput_match = re.search(
        r"^\s*([0-9.]+)s for (\d+) batches -> ([0-9.]+) batches/s \(([0-9,]+) positions/s\)$",
        stdout,
        re.MULTILINE,
    )
    if throughput_match:
        result.elapsed_s = float(throughput_match.group(1))
        result.timed_batches = int(throughput_match.group(2))
        result.batches_per_s = float(throughput_match.group(3))
        result.positions_per_s = _parse_int(throughput_match.group(4))

    rates_match = re.search(
        r"^timed_rates decoded=([^ ]+) skipped=([^ ]+) encoded=([^ ]+)$",
        stdout,
        re.MULTILINE,
    )
    if rates_match:
        result.decoded_per_s = _parse_rate(rates_match.group(1))
        result.skipped_per_s = _parse_rate(rates_match.group(2))
        result.encoded_per_s = _parse_rate(rates_match.group(3))

    ratios_match = re.search(
        r"^timed_ratios keep=([0-9.]+)% skip=([0-9.]+)%$",
        stdout,
        re.MULTILINE,
    )
    if ratios_match:
        result.keep_ratio_pct = float(ratios_match.group(1))
        result.skip_ratio_pct = float(ratios_match.group(2))

    return result


def _run_one(
    cfg: SweepConfig, total_threads: int, decode_threads: int, encode_threads: int
) -> SweepResult:
    proc = subprocess.run(
        _build_command(cfg, total_threads, decode_threads, encode_threads),
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        timeout=cfg.timeout_seconds,
    )
    return _parse_output(total_threads, decode_threads, encode_threads, proc)


def _write_csv(path: str, results: list[SweepResult]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def _print_run_summary(result: SweepResult) -> None:
    status = "ok" if result.returncode == 0 else f"fail({result.returncode})"
    print(
        f"total={result.total_threads:<4} "
        f"decode={result.decode_threads:<4} "
        f"encode={result.encode_threads:<4} "
        f"first={_format_float(result.first_batch_latency_s, 3):>7}s "
        f"batch/s={_format_float(result.batches_per_s):>7} "
        f"pos/s={_format_int(result.positions_per_s):>10} "
        f"decoded/s={_format_float(result.decoded_per_s, 0):>10} "
        f"keep%={_format_float(result.keep_ratio_pct, 3):>7} "
        f"status={status}"
    )


def _print_ranking(results: list[SweepResult]) -> None:
    successes = [
        r for r in results if r.returncode == 0 and r.positions_per_s is not None
    ]
    if not successes:
        print("No successful runs to rank.")
        return

    ranked = sorted(successes, key=lambda r: r.positions_per_s or -1, reverse=True)
    print("\nBest Results")
    for idx, result in enumerate(ranked[: min(12, len(ranked))], start=1):
        print(
            f"{idx:>2}. total={result.total_threads} "
            f"decode={result.decode_threads} "
            f"encode={result.encode_threads} "
            f"pos/s={_format_int(result.positions_per_s)} "
            f"batch/s={_format_float(result.batches_per_s)} "
            f"decoded/s={_format_float(result.decoded_per_s, 0)} "
            f"keep%={_format_float(result.keep_ratio_pct, 3)} "
            f"first={_format_float(result.first_batch_latency_s, 3)}s"
        )


def main() -> None:
    cfg = tyro.cli(SweepConfig)
    if not BENCH_SCRIPT.is_file():
        raise SystemExit(f"missing benchmark script: {BENCH_SCRIPT}")
    if not hasattr(os, "sched_getaffinity"):
        raise SystemExit("this sweep script requires os.sched_getaffinity on Linux")

    affinity = len(os.sched_getaffinity(0))
    cases = _cases(cfg, affinity)
    if not cases:
        raise SystemExit("no sweep cases generated")

    totals = sorted({total for total, _, _ in cases})
    print(
        f"cpu_affinity={affinity} totals={totals} decode_percents={cfg.decode_percents} cases={len(cases)} python={sys.executable}",
        flush=True,
    )

    results: list[SweepResult] = []
    for total_threads, decode_threads, encode_threads in cases:
        print(
            f"\n=== sweep total={total_threads} decode={decode_threads} encode={encode_threads} ===",
            flush=True,
        )
        try:
            result = _run_one(cfg, total_threads, decode_threads, encode_threads)
        except subprocess.TimeoutExpired as exc:
            result = SweepResult(
                total_threads=total_threads,
                decode_threads=decode_threads,
                encode_threads=encode_threads,
                returncode=124,
                stdout=_coerce_text(exc.stdout),
                stderr=_coerce_text(exc.stderr)
                or f"timed out after {cfg.timeout_seconds}s",
            )

        results.append(result)
        _print_run_summary(result)

        if cfg.show_full_output or result.returncode != 0:
            if result.stdout:
                print("--- stdout ---")
                print(result.stdout.rstrip())
            if result.stderr:
                print("--- stderr ---")
                print(result.stderr.rstrip())

        if cfg.stop_on_error and result.returncode != 0:
            break

    print()
    _print_ranking(results)

    if cfg.output_csv and results:
        _write_csv(cfg.output_csv, results)
        print(f"\nWrote CSV to {cfg.output_csv}")


if __name__ == "__main__":
    main()
