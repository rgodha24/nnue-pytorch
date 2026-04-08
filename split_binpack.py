#!/usr/bin/env python3
"""Split a binpack file into N chunks for parallel loading.

Binpack format:
  - Each chunk: "BINP" magic (4 bytes) + chunkSize (4 bytes, little-endian) + compressed_data
  - We split by copying entire chunks to different output files
"""

import sys
import struct
import argparse
from pathlib import Path


def count_chunks(filepath):
    """Count total chunks in binpack file."""
    count = 0
    with open(filepath, "rb") as f:
        while True:
            # Read magic
            magic = f.read(4)
            if len(magic) < 4:
                break
            if magic != b"BINP":
                raise ValueError(f"Invalid magic: {magic!r}")

            # Read size
            size_bytes = f.read(4)
            if len(size_bytes) < 4:
                break
            chunk_size = struct.unpack("<I", size_bytes)[0]

            # Skip chunk data
            f.seek(chunk_size, 1)
            count += 1

            if count % 1000 == 0:
                print(f"  Counted {count} chunks...", end="\r")

    return count


def split_binpack(input_path, num_chunks):
    """Split binpack into N roughly equal files."""
    input_path = Path(input_path)

    print(f"Counting chunks in {input_path}...")
    total_chunks = count_chunks(input_path)
    print(f"Total chunks: {total_chunks}")

    chunks_per_file = total_chunks // num_chunks
    print(f"Splitting into {num_chunks} files (~{chunks_per_file} chunks each)")

    # Open output files
    outputs = []
    for i in range(num_chunks):
        output_path = input_path.with_suffix(f".part{i:03d}.binpack")
        outputs.append(open(output_path, "wb"))
        print(f"  {output_path}")

    # Distribute chunks round-robin
    with open(input_path, "rb") as f:
        chunk_idx = 0
        while True:
            # Read magic
            magic = f.read(4)
            if len(magic) < 4:
                break

            # Read size
            size_bytes = f.read(4)
            if len(size_bytes) < 4:
                break
            chunk_size = struct.unpack("<I", size_bytes)[0]

            # Read chunk data
            data = f.read(chunk_size)
            if len(data) < chunk_size:
                break

            # Write to appropriate output
            output_idx = chunk_idx % num_chunks
            outputs[output_idx].write(magic)
            outputs[output_idx].write(size_bytes)
            outputs[output_idx].write(data)

            chunk_idx += 1
            if chunk_idx % 1000 == 0:
                print(f"  Processed {chunk_idx}/{total_chunks} chunks...", end="\r")

    # Close outputs
    for f in outputs:
        f.close()

    print(f"\nDone! Split {chunk_idx} chunks into {num_chunks} files")

    # Print sizes
    for i in range(num_chunks):
        output_path = input_path.with_suffix(f".part{i:03d}.binpack")
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  {output_path.name}: {size_mb:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split binpack file into chunks")
    parser.add_argument("input", help="Input binpack file")
    parser.add_argument(
        "-n", "--num-chunks", type=int, default=8, help="Number of chunks (default: 8)"
    )

    args = parser.parse_args()

    split_binpack(args.input, args.num_chunks)
