import math
import unittest

import numpy as np
import torch

from context_maskgit_v2 import ContextMaskGITv2
from context_maskgit_v3 import (
    ContextMaskGITv3,
    initialize_context_maskgit_v3_from_v2,
    load_context_maskgit_v3,
)
from tokcom_arq import analytical_arq_metrics, transmit_tokens_with_arq


class ContextMaskGITv3Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(1501)

    def test_context_gradients_dropout_and_recovery_invariants(self):
        model = ContextMaskGITv3(d=16, layers=1, local_blocks=1, classes=10, context_dropout=0.5)
        model.train()
        ids = torch.randint(0, 1024, (4, 256))
        labels = torch.arange(4)
        loss, missing, used_labels = model.loss(ids, labels, per=0.3)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(float(model.condition.weight.grad.abs().sum()), 0.0)
        self.assertEqual(used_labels.shape, labels.shape)
        model.eval()
        recovered, confidence = model.recover_with_confidence(
            ids.masked_fill(missing, 1024), missing, labels, steps=3, guidance_scale=1.5
        )
        self.assertTrue(torch.equal(recovered[~missing], ids[~missing]))
        self.assertTrue(bool(((confidence >= 0) & (confidence <= 1)).all()))

    def test_classifier_free_guidance_endpoints_and_checkpoint_loader(self):
        model = ContextMaskGITv3(d=16, layers=1, local_blocks=1, classes=10).eval()
        ids = torch.randint(0, 1025, (2, 256))
        labels = torch.tensor([1, 7])
        self.assertTrue(torch.allclose(model.guided_logits(ids, labels, 0), model(ids, None)))
        self.assertTrue(torch.allclose(model.guided_logits(ids, labels, 1), model(ids, labels)))
        checkpoint = {"model": "context_maskgit_v3", "config": model.config, "state": model.state_dict()}
        loaded = load_context_maskgit_v3(checkpoint)
        self.assertTrue(torch.allclose(loaded(ids, labels), model(ids, labels)))

    def test_v2_warm_start_preserves_unconditional_logits(self):
        v2 = ContextMaskGITv2(d=16, layers=1, local_blocks=1).eval()
        checkpoint = {"model": "context_maskgit_v2", "config": v2.config, "state": v2.state_dict()}
        v3 = initialize_context_maskgit_v3_from_v2(checkpoint, classes=10).eval()
        ids = torch.randint(0, 1025, (2, 256))
        self.assertTrue(torch.allclose(v2(ids), v3(ids, None), atol=0, rtol=0))

    def test_analytical_tce_matches_paper_definition(self):
        result = analytical_arq_metrics(0.2)
        self.assertAlmostEqual(result["expected_transmissions_per_packet"], 1.25)
        self.assertAlmostEqual(result["source_bpp"], 2560 / 65536)
        self.assertAlmostEqual(result["tce_pixels_per_token_bit"], 65536 / (1.25 * 2560))
        with self.assertRaises(ValueError):
            analytical_arq_metrics(1.0)

    def test_noiseless_arq_preserves_tokens_without_retransmission(self):
        ids = np.arange(16, dtype=np.int64).reshape(1, 16)
        restored, missing, metrics = transmit_tokens_with_arq(
            ids, channel="noiseless", snr_db=100, packet_tokens=4, max_rounds=3
        )
        np.testing.assert_array_equal(restored, ids)
        self.assertFalse(missing.any())
        self.assertEqual(metrics["packet_attempts"], 4)
        self.assertEqual(metrics["retransmission_packets"], 0)
        self.assertEqual(metrics["token_error_rate"], 0)

    def test_confidence_policy_can_decline_retransmission(self):
        model = ContextMaskGITv3(d=16, layers=1, local_blocks=1, classes=10).eval()
        ids = (np.arange(256, dtype=np.int64) % 1024).reshape(1, 256)
        labels = torch.tensor([0])
        _, _, metrics = transmit_tokens_with_arq(
            ids,
            receiver=model,
            labels=labels,
            snr_db=-20,
            channel="awgn",
            seed=7,
            packet_tokens=4,
            max_rounds=4,
            policy="confidence",
            confidence_threshold=0.0,
            recovery_steps=2,
        )
        self.assertEqual(metrics["retransmission_packets"], 0)
        self.assertEqual(metrics["packet_attempts"], metrics["initial_packets"])
        self.assertTrue(math.isfinite(metrics["tce_pixels_per_token_bit"]))


if __name__ == "__main__":
    unittest.main()
