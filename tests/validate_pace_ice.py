#!/usr/bin/env python3
"""
PACE ICE validation script for the Rust loader close() fix.
Run this on the PACE ICE node to verify the race condition is fixed.
"""

import gc
import os
import sys
import time
import signal
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nnue_loader


def alarm_handler(signum, frame):
    """Handle timeout alarm."""
    raise TimeoutError("Test timed out - race condition still present!")


def test_close_with_live_batches():
    """Test the main race condition fix."""
    print("Test 1: close() with live batches...", end=" ")

    # Find a test binpack (check PACE ICE locations first)
    test_paths = [
        os.path.expanduser(
            "~/scratch/nnue-data/T60T70wIsRightFarseerT60T74T75T76.split_0.binpack"
        ),
        os.path.expanduser(
            "~/scratch/nnue-data/multinet_pv-2_diff-100_nodes-5000.binpack"
        ),
        os.path.expanduser("~/scratch/nnue-data/nodes5000pv2_UHO.binpack"),
        os.path.expanduser("~/scratch/nnue-data/dfrc_n5000.binpack"),
        "/home/rgodha/Developer/knightfall/nnue-pytorch/.pgo/small.binpack",
        "/tmp/nnue-data/wrongIsRight_nodes5000pv2.binpack",
        "/tmp/nnue-data/nodes5000pv2_UHO.binpack",
    ]

    binpack_path = None
    for path in test_paths:
        if os.path.exists(path):
            binpack_path = path
            break

    if not binpack_path:
        print("SKIP (no test data)")
        return True

    # Set a 10-second alarm - if we hang, this will fire
    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(10)

    try:
        stream = nnue_loader.BatchStream(
            feature_set="HalfKAv2_hm",
            filenames=[binpack_path],
            batch_size=32,
            total_threads=4,
            shuffle_buffer_entries=0,
        )

        # Fetch batches and KEEP THEM ALIVE (this was the race condition)
        batches = []
        for _ in range(5):
            batch = stream.next_batch()
            if batch is None:
                break
            batches.append(batch)

        # This is where the hang would occur before the fix
        stream.close()

        # Verify batches are still accessible
        for batch in batches:
            _ = batch["white"]

        signal.alarm(0)  # Cancel alarm
        print("PASS")
        return True

    except TimeoutError:
        print("FAIL - close() hung for >10 seconds!")
        return False
    except Exception as e:
        signal.alarm(0)
        print(f"FAIL - {e}")
        return False


def test_stress_close_iterations():
    """Stress test: open/fetch/close many times."""
    print("Test 2: Stress test (10 iterations)...", end=" ")

    test_paths = [
        os.path.expanduser(
            "~/scratch/nnue-data/T60T70wIsRightFarseerT60T74T75T76.split_0.binpack"
        ),
        os.path.expanduser(
            "~/scratch/nnue-data/multinet_pv-2_diff-100_nodes-5000.binpack"
        ),
        os.path.expanduser("~/scratch/nnue-data/nodes5000pv2_UHO.binpack"),
        os.path.expanduser("~/scratch/nnue-data/dfrc_n5000.binpack"),
        "/home/rgodha/Developer/knightfall/nnue-pytorch/.pgo/small.binpack",
        "/tmp/nnue-data/wrongIsRight_nodes5000pv2.binpack",
        "/tmp/nnue-data/nodes5000pv2_UHO.binpack",
    ]

    binpack_path = None
    for path in test_paths:
        if os.path.exists(path):
            binpack_path = path
            break

    if not binpack_path:
        print("SKIP (no test data)")
        return True

    signal.signal(signal.SIGALRM, alarm_handler)

    try:
        for i in range(10):
            signal.alarm(10)

            stream = nnue_loader.BatchStream(
                feature_set="HalfKAv2_hm",
                filenames=[binpack_path],
                batch_size=32,
                total_threads=4,
                shuffle_buffer_entries=0,
            )

            # Fetch some batches
            batches = []
            for _ in range(3):
                batch = stream.next_batch()
                if batch:
                    batches.append(batch)

            # Close (potentially with live batches)
            stream.close()

            # Force GC to clean up
            del batches
            gc.collect()

            signal.alarm(0)

        print("PASS")
        return True

    except TimeoutError:
        print("FAIL - hung on iteration!")
        return False
    except Exception as e:
        signal.alarm(0)
        print(f"FAIL - {e}")
        return False


def test_gc_pressure():
    """Test that GC doesn't cause issues with close()."""
    print("Test 3: GC pressure test...", end=" ")

    test_paths = [
        os.path.expanduser(
            "~/scratch/nnue-data/T60T70wIsRightFarseerT60T74T75T76.split_0.binpack"
        ),
        os.path.expanduser(
            "~/scratch/nnue-data/multinet_pv-2_diff-100_nodes-5000.binpack"
        ),
        os.path.expanduser("~/scratch/nnue-data/nodes5000pv2_UHO.binpack"),
        os.path.expanduser("~/scratch/nnue-data/dfrc_n5000.binpack"),
        "/home/rgodha/Developer/knightfall/nnue-pytorch/.pgo/small.binpack",
        "/tmp/nnue-data/wrongIsRight_nodes5000pv2.binpack",
        "/tmp/nnue-data/nodes5000pv2_UHO.binpack",
    ]

    binpack_path = None
    for path in test_paths:
        if os.path.exists(path):
            binpack_path = path
            break

    if not binpack_path:
        print("SKIP (no test data)")
        return True

    signal.signal(signal.SIGALRM, alarm_handler)

    try:
        signal.alarm(10)

        stream = nnue_loader.BatchStream(
            feature_set="HalfKAv2_hm",
            filenames=[binpack_path],
            batch_size=32,
            total_threads=4,
            shuffle_buffer_entries=0,
        )

        # Fetch batches
        batches = []
        for _ in range(5):
            batch = stream.next_batch()
            if batch:
                batches.append(batch)

        # Run aggressive GC
        for _ in range(5):
            gc.collect()

        # Close should still work
        stream.close()

        signal.alarm(0)
        print("PASS")
        return True

    except TimeoutError:
        print("FAIL - hung after GC!")
        return False
    except Exception as e:
        signal.alarm(0)
        print(f"FAIL - {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("PACE ICE Rust Loader Close() Fix Validation")
    print("=" * 60)
    print()

    results = []

    results.append(("Close with live batches", test_close_with_live_batches()))
    results.append(("Stress close iterations", test_stress_close_iterations()))
    results.append(("GC pressure", test_gc_pressure()))

    print()
    print("=" * 60)
    print("Results:")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All tests PASSED! The race condition fix is working.")
        return 0
    else:
        print("Some tests FAILED! The race condition may still be present.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
