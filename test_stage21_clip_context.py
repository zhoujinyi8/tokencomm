import hashlib
import json
import unittest
from pathlib import Path

import torch

from build_stage21_clip_context import PROMPT_TEMPLATE, class_prompt, sha256_file, sha256_json, sha256_tensor


class Stage21ClipContextTests(unittest.TestCase):
    def test_prompt_is_readable_and_frozen(self):
        self.assertEqual(class_prompt("macaroni_and_cheese"), "a photo of macaroni and cheese, a type of food")
        self.assertEqual(PROMPT_TEMPLATE.count("{}"), 1)

    def test_class_hash_is_order_sensitive(self):
        self.assertNotEqual(sha256_json(["a", "b"]), sha256_json(["b", "a"]))

    def test_tensor_hash_is_raw_content_hash(self):
        value = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        self.assertEqual(sha256_tensor(value), hashlib.sha256(value.numpy().tobytes()).hexdigest())

    def test_generated_embeddings_align_with_food101_labels(self):
        root = Path(__file__).resolve().parent
        source = json.loads((root / "data/stage20_food101/manifest.json").read_text(encoding="utf-8"))
        audit = json.loads((root / "STAGE21_CLIP_CONTEXT_MANIFEST.json").read_text(encoding="utf-8"))
        output = torch.load(root / audit["embedding_file"], map_location="cpu", weights_only=True)
        self.assertEqual(output["class_names"], source["classes"])
        self.assertEqual(tuple(output["embeddings"].shape), (101, 512))
        self.assertTrue(torch.allclose(output["embeddings"].norm(dim=-1), torch.ones(101), atol=1e-6, rtol=0))
        self.assertEqual(audit["embedding_file_sha256"], sha256_file(root / audit["embedding_file"]))
        self.assertEqual(audit["embedding_tensor_sha256"], sha256_tensor(output["embeddings"]))
        self.assertFalse(audit["test_accessed"])


if __name__ == "__main__":
    unittest.main()
