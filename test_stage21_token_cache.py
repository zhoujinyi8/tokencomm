import unittest
import json
from pathlib import Path

import torch

from cache_stage21_food101_tokens import OUTPUT, expected_shards, validate_protocol, validate_shard
from finalize_stage21_token_cache import finalize


class Stage21TokenCacheTests(unittest.TestCase):
    def test_shard_count_handles_partial_tail(self):
        self.assertEqual(expected_shards(64, 32), 2)
        self.assertEqual(expected_shards(65, 32), 3)
        self.assertEqual(expected_shards(1, 32), 1)

    def test_frozen_protocol_and_first_real_shard(self):
        root = Path(__file__).resolve().parent
        protocol = json.loads((root / "STAGE21_TOKEN_CACHE_PROTOCOL.json").read_text(encoding="utf-8"))
        identity = validate_protocol(protocol)
        self.assertEqual(protocol["splits"], {"train": 70700, "dev": 5050})
        self.assertFalse(protocol["test_accessed"])
        shard = torch.load(OUTPUT / "train/00000.pt", map_location="cpu", weights_only=True)
        validate_shard(shard, identity, "train", 0, 32)

    def test_aggregate_is_fail_closed_while_cache_is_partial(self):
        with self.assertRaises(RuntimeError):
            finalize()


if __name__ == "__main__":
    unittest.main()
