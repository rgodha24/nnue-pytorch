#!/usr/bin/env python3
"""Profile the dataloader and model to find bottlenecks - uses defaults."""

import sys
import time

import torch

sys.path.insert(0, "data_loader")

import data_loader


def profile_dataloader(filenames, num_batches=200):
    """Profile the dataloader using default config."""
    print(f"\n=== Dataloader Profile ===")

    # C++ dataloader uses INPUT_FEATURE_NAME (without ^)
    dataset = data_loader.SparseBatchDataset(
        "Full_Threats+HalfKAv2_hm",
        filenames,
        batch_size=16384,
    )

    finite_dataset = data_loader.FixedNumBatchesDataset(
        dataset,
        num_batches,
        pin_memory=True,
    )

    loader = torch.utils.data.DataLoader(
        finite_dataset,
        batch_size=None,
        num_workers=0,
    )

    times = []
    it = iter(loader)
    for i in range(num_batches):
        t0 = time.perf_counter()
        batch = next(it)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        if i % 50 == 0:
            throughput = (
                16384 / (sum(times[-10:]) / len(times[-10:])) / 1e6
                if len(times) >= 10
                else 0
            )
            print(
                f"  Batch {i}/{num_batches}: {times[-1] * 1000:.2f} ms ({throughput:.2f} MPos/s)"
            )

    avg_time = sum(times) / len(times)
    throughput = 16384 / avg_time / 1e6

    print(f"\nResults:")
    print(f"  Avg batch time: {avg_time * 1000:.2f} ms")
    print(f"  Throughput: {throughput:.2f} MPos/s")
    print(f"  Batches/s: {1 / avg_time:.1f}")

    return throughput


def profile_combined(filenames, num_batches=100):
    """Profile dataloader + GPU forward pass."""
    print(f"\n=== Combined Profile ===")

    from model.lightning_module import NNUE
    from model.config import NNUELightningConfig

    config = NNUELightningConfig()
    model = NNUE(config=config, max_epoch=1, num_batches_per_epoch=100)
    model = model.cuda()
    model.eval()

    # C++ dataloader uses INPUT_FEATURE_NAME (without ^)
    dataset = data_loader.SparseBatchDataset(
        model.model.input_feature_name,
        filenames,
        batch_size=16384,
    )

    finite_dataset = data_loader.FixedNumBatchesDataset(
        dataset,
        num_batches,
        pin_memory=True,
    )

    loader = torch.utils.data.DataLoader(
        finite_dataset,
        batch_size=None,
        num_workers=0,
    )

    torch.cuda.synchronize()
    print("  Warming up...")
    it = iter(loader)
    for _ in range(5):
        batch = next(it)
        # model takes: us, them, white_indices, white_values, black_indices, black_values, psqt_indices, layer_stack_indices
        # dataloader returns 10 tensors including outcome and score for loss computation
        batch_gpu = [b.cuda(non_blocking=True) for b in batch]
        _ = model.model(*batch_gpu[:6], batch_gpu[8], batch_gpu[9])

    torch.cuda.synchronize()
    print("  Starting benchmark...")

    dataloader_times = []
    gpu_times = []
    total_times = []

    for i in range(num_batches):
        t0 = time.perf_counter()

        batch = next(it)
        t1 = time.perf_counter()

        batch_gpu = [b.cuda(non_blocking=True) for b in batch]
        _ = model.model(*batch_gpu[:6], batch_gpu[8], batch_gpu[9])
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        dataloader_times.append(t1 - t0)
        gpu_times.append(t2 - t1)
        total_times.append(t2 - t0)

        if i % 25 == 0:
            print(
                f"  Batch {i}/{num_batches}: dl={dataloader_times[-1] * 1000:.1f}ms gpu={gpu_times[-1] * 1000:.1f}ms"
            )

    print(f"\nResults:")
    print(
        f"  Dataloader avg: {sum(dataloader_times) / len(dataloader_times) * 1000:.2f} ms"
    )
    print(f"  GPU avg: {sum(gpu_times) / len(gpu_times) * 1000:.2f} ms")
    print(f"  Total avg: {sum(total_times) / len(total_times) * 1000:.2f} ms")
    print(
        f"  Combined throughput: {16384 / (sum(total_times) / len(total_times)) / 1e6:.2f} MPos/s"
    )


def main():
    filenames = ["nnue-data/nodes5000pv2_UHO.binpack"]

    print("=" * 60)
    print("NNUE PYTORCH PERFORMANCE PROFILING")
    print("=" * 60)

    if torch.cuda.is_available():
        print(f"\nGPU: {torch.cuda.get_device_name()}")
        print(f"CUDA Version: {torch.version.cuda}")
        profile_combined(filenames)

    profile_dataloader(filenames)


if __name__ == "__main__":
    main()
