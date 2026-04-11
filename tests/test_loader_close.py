"""End-to-end tests for the Rust loader close() behavior."""

import gc
import os
import sys
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nnue_loader


# Use small test file from PGO data
TEST_BINPACK = "/home/rgodha/Developer/knightfall/nnue-pytorch/.pgo/small.binpack"
# Fallback to real data if small doesn't exist
REAL_BINPACK = "/tmp/nnue-data/wrongIsRight_nodes5000pv2.binpack"


def get_test_file():
    """Get a test binpack file."""
    if os.path.exists(TEST_BINPACK):
        return TEST_BINPACK
    if os.path.exists(REAL_BINPACK):
        return REAL_BINPACK
    raise FileNotFoundError("No test binpack file found")


class TestLoaderCloseBehavior(unittest.TestCase):
    """Test that stream.close() doesn't hang when batches are still alive."""

    def test_close_with_live_batches(self):
        """Test that close() works even when batches are held in Python."""
        binpack_path = get_test_file()

        stream = nnue_loader.BatchStream(
            feature_set="HalfKAv2_hm",
            filenames=[binpack_path],
            batch_size=32,
            total_threads=2,
            shuffle_buffer_entries=0,
        )

        # Fetch some batches but keep them alive
        batches = []
        for _ in range(3):
            batch = stream.next_batch()
            self.assertIsNotNone(batch)
            batches.append(batch)

        # Now close the stream while batches are still alive
        # This should NOT hang
        stream.close()

        # Batches should still be accessible
        for batch in batches:
            self.assertIn("white", batch)
            self.assertIn("black", batch)

    def test_close_timeout(self):
        """Test that close() completes within reasonable time."""
        binpack_path = get_test_file()

        stream = nnue_loader.BatchStream(
            feature_set="HalfKAv2_hm",
            filenames=[binpack_path],
            batch_size=32,
            total_threads=4,
            shuffle_buffer_entries=0,
        )

        # Fetch batches and keep them alive
        batches = []
        for _ in range(5):
            batch = stream.next_batch()
            if batch is None:
                break
            batches.append(batch)

        # Try to close with a timeout
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(stream.close)
            try:
                future.result(timeout=5.0)  # Should complete within 5 seconds
            except TimeoutError:
                self.fail(
                    "stream.close() timed out - workers likely stuck waiting for free_rx"
                )

    def test_multiple_close_calls(self):
        """Test that multiple close() calls don't cause issues."""
        binpack_path = get_test_file()

        stream = nnue_loader.BatchStream(
            feature_set="HalfKAv2_hm",
            filenames=[binpack_path],
            batch_size=32,
            total_threads=2,
            shuffle_buffer_entries=0,
        )

        # Fetch a batch
        batch = stream.next_batch()
        self.assertIsNotNone(batch)

        # Close multiple times
        stream.close()
        stream.close()
        stream.close()

        # Batch should still be valid
        self.assertIn("white", batch)

    def test_gc_before_close(self):
        """Test behavior when GC runs before close with live batches."""
        binpack_path = get_test_file()

        stream = nnue_loader.BatchStream(
            feature_set="HalfKAv2_hm",
            filenames=[binpack_path],
            batch_size=32,
            total_threads=2,
            shuffle_buffer_entries=0,
        )

        # Fetch batches
        batches = []
        for _ in range(3):
            batch = stream.next_batch()
            self.assertIsNotNone(batch)
            batches.append(batch)

        # Run GC while batches are alive
        gc.collect()

        # Close should still work
        stream.close()


class TestLoaderBasicFunctionality(unittest.TestCase):
    """Basic functionality tests for the Rust loader."""

    def test_stream_iteration(self):
        """Test that we can iterate through a stream."""
        binpack_path = get_test_file()

        stream = nnue_loader.BatchStream(
            feature_set="HalfKAv2_hm",
            filenames=[binpack_path],
            batch_size=32,
            total_threads=2,
            shuffle_buffer_entries=0,
        )

        batch_count = 0
        for batch in stream:
            self.assertIn("white", batch)
            self.assertIn("black", batch)
            self.assertIn("is_white", batch)
            self.assertIn("outcome", batch)
            self.assertIn("score", batch)
            batch_count += 1
            if batch_count >= 2:
                break

        self.assertGreaterEqual(batch_count, 1)

    def test_stream_stats(self):
        """Test that stream stats work correctly."""
        binpack_path = get_test_file()

        stream = nnue_loader.BatchStream(
            feature_set="HalfKAv2_hm",
            filenames=[binpack_path],
            batch_size=32,
            total_threads=2,
            shuffle_buffer_entries=0,
        )

        # Get initial stats
        stats = stream.stats()
        self.assertIn("decoded_entries", stats)
        self.assertIn("encoded_entries", stats)
        self.assertIn("produced_batches", stats)

        # Fetch some batches
        for _ in range(3):
            batch = stream.next_batch()
            if batch is None:
                break

        # Get updated stats
        stats = stream.stats()
        self.assertGreater(stats["decoded_entries"], 0)
        self.assertGreater(stats["encoded_entries"], 0)

        stream.close()


if __name__ == "__main__":
    unittest.main()
