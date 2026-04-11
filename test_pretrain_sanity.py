#!/usr/bin/env python3
"""Pre-training sanity check - tests loader, model, and training setup."""

import sys
import torch
import lightning as L
from torch.utils.data import DataLoader

# Add repo to path
sys.path.insert(0, "/home/hice1/rgodha3/nnue-pytorch")

import data_loader
import model as M
from config import TrainingConfig
from data_loader.config import DataloaderSkipConfig

print("=" * 60)
print("PRE-TRAINING SANITY CHECK")
print("=" * 60)

# 1. Test loader initialization
print("\n1. Testing data loader initialization...")
try:
    dataset = data_loader.SparseBatchDataset(
        feature_set="Full_Threats+HalfKAv2_hm^",
        filenames=[
            "/home/hice1/rgodha3/scratch/nnue-data/T60T70wIsRightFarseerT60T74T75T76.split_0.binpack"
        ],
        batch_size=65536,
        num_workers=1,
        config=DataloaderSkipConfig(
            filtered=True,
            random_fen_skipping=10,
            early_fen_skipping=12,
        ),
    )
    print("✅ SparseBatchDataset created")
except Exception as e:
    print(f"❌ Failed to create dataset: {e}")
    sys.exit(1)

# 2. Test FixedNumBatchesDataset wrapper
print("\n2. Testing FixedNumBatchesDataset wrapper...")
try:
    fixed_ds = data_loader.FixedNumBatchesDataset(
        dataset,
        num_batches=5,
        pin_memory=False,
    )
    print("✅ FixedNumBatchesDataset created")
except Exception as e:
    print(f"❌ Failed to create FixedNumBatchesDataset: {e}")
    sys.exit(1)

# 3. Test DataLoader wrapper
print("\n3. Testing PyTorch DataLoader...")
try:
    loader = DataLoader(
        fixed_ds,
        batch_size=None,
        batch_sampler=None,
        num_workers=0,
    )
    print("✅ DataLoader created")
except Exception as e:
    print(f"❌ Failed to create DataLoader: {e}")
    sys.exit(1)

# 4. Test fetching a batch
print("\n4. Fetching first batch...")
try:
    import time

    start = time.time()
    batch = next(iter(loader))
    elapsed = time.time() - start
    print(f"✅ Got batch in {elapsed:.2f}s")
    print(f"   Batch has {len(batch)} tensors")
    print(f"   is_white shape: {batch[0].shape}")
except Exception as e:
    print(f"❌ Failed to fetch batch: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# 5. Test model creation
print("\n5. Testing model creation...")
try:
    nnue = M.NNUE(
        num_inputs=768,
        feature_set=M.FeatureSet.Full_Threats_HalfKAv2_hm,
        layer_counts=(1024, 31),
        activation_fn=M.ActivationFn.SCReLU,
    )
    print(
        f"✅ Model created: {sum(p.numel() for p in nnue.parameters()) / 1e6:.1f}M parameters"
    )
except Exception as e:
    print(f"❌ Failed to create model: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# 6. Test forward pass
print("\n6. Testing forward pass...")
try:
    nnue = nnue.cuda()
    # Use batch from above
    (
        us,
        them,
        white_idx,
        white_val,
        black_idx,
        black_val,
        outcome,
        score,
        psqt_idx,
        layer_stack_idx,
    ) = batch

    us = us.cuda()
    them = them.cuda()
    white_idx = white_idx.cuda()
    white_val = white_val.cuda()
    black_idx = black_idx.cuda()
    black_val = black_val.cuda()
    outcome = outcome.cuda()
    score = score.cuda()
    psqt_idx = psqt_idx.cuda()
    layer_stack_idx = layer_stack_idx.cuda()

    start = time.time()
    output = nnue(
        (us, them),
        (white_idx, white_val),
        (black_idx, black_val),
        psqt_idx,
        layer_stack_idx,
    )
    elapsed = time.time() - start
    print(f"✅ Forward pass successful in {elapsed:.3f}s")
    print(f"   Output shape: {output.shape}")
except Exception as e:
    print(f"❌ Forward pass failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# 7. Test backward pass
print("\n7. Testing backward pass...")
try:
    loss = output.mean()
    start = time.time()
    loss.backward()
    elapsed = time.time() - start
    print(f"✅ Backward pass successful in {elapsed:.3f}s")
except Exception as e:
    print(f"❌ Backward pass failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("ALL CHECKS PASSED! ✅")
print("Training should work now.")
print("=" * 60)
