import os
import unittest
from unittest.mock import patch

import train
from data_loader.config import DataloaderDDPConfig, DataloaderSkipConfig


class TestTrainDDPHelpers(unittest.TestCase):
    def test_num_batches_for_size_uses_global_batch(self):
        self.assertEqual(train.num_batches_for_size(100_000_000, 65_536), 1526)
        self.assertEqual(train.num_batches_for_size(0, 65_536), 0)

    def test_resolve_dataloader_ddp_config_uses_env_world_size(self):
        with patch.dict(os.environ, {"WORLD_SIZE": "4", "RANK": "2"}, clear=True):
            cfg = train.resolve_dataloader_ddp_config(2)

        self.assertEqual(cfg, DataloaderDDPConfig(rank=2, world_size=4))

    def test_resolve_dataloader_ddp_config_defaults_to_local_devices(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = train.resolve_dataloader_ddp_config(2)

        self.assertEqual(cfg, DataloaderDDPConfig(rank=0, world_size=2))

    def test_make_data_loaders_passes_ddp_config_and_global_batch(self):
        ddp_config = DataloaderDDPConfig(rank=1, world_size=2)
        skip_config = DataloaderSkipConfig(
            filtered=False,
            wld_filtered=False,
            random_fen_skipping=0,
            early_fen_skipping=-1,
            simple_eval_skipping=-1,
        )

        train_loader, val_loader = train.make_data_loaders(
            ["train.binpack"],
            ["val.binpack"],
            "HalfKAv2_hm",
            num_workers=1,
            batch_size=32,
            global_batch_size=64,
            config=skip_config,
            epoch_size=1000,
            val_size=128,
            pin_memory=False,
            queue_size_limit=4,
            ddp_config=ddp_config,
        )

        self.assertEqual(
            len(train_loader.dataset), train.num_batches_for_size(1000, 64)
        )
        self.assertEqual(len(val_loader.dataset), train.num_batches_for_size(128, 64))
        self.assertEqual(train_loader.dataset.dataset.ddp_config, ddp_config)
        self.assertEqual(val_loader.dataset.dataset.ddp_config, ddp_config)
        self.assertEqual(train_loader.dataset.dataset.batch_size, 32)
        self.assertEqual(val_loader.dataset.dataset.batch_size, 32)


if __name__ == "__main__":
    unittest.main()
