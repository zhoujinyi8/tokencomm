import unittest

import torch

from context_maskgit_v3 import load_context_maskgit_v3
from context_maskgit_v4 import ContextMaskGITv4, initialize_context_maskgit_v4_from_v3, load_context_maskgit_v4


class ContextMaskGITv4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checkpoint = torch.load("runs/stage15_contextual_tokcom/best.pt", weights_only=True)

    def test_v3_warm_start_preserves_conditional_and_unconditional_logits(self):
        v3 = load_context_maskgit_v3(self.checkpoint).eval()
        v4 = initialize_context_maskgit_v4_from_v3(self.checkpoint).eval()
        ids = torch.randint(0, 1024, (2, 256))
        labels = torch.tensor([1, 7])
        with torch.no_grad():
            self.assertTrue(torch.equal(v3(ids, labels), v4(ids, labels)))
            self.assertTrue(torch.equal(v3(ids, None), v4(ids, None)))

    def test_new_condition_paths_receive_gradients_and_round_trip(self):
        model = ContextMaskGITv4(d=32, layers=1, local_blocks=1, classes=3, context_prior_weight=0.2)
        ids = torch.randint(0, 1024, (2, 256))
        labels = torch.tensor([0, 2])
        total, missing, _, parts = model.loss_components(ids, labels, per=0.3)
        total.backward()
        self.assertTrue(bool(missing.any()))
        self.assertGreater(float(model.condition_film.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.condition_prior.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.condition.weight.grad.abs().sum()), 0)
        self.assertTrue(torch.isfinite(parts["masked_ce"]))
        self.assertTrue(torch.isfinite(parts["context_prior_ce"]))
        checkpoint = {"model": "context_maskgit_v4", "config": model.config, "state": model.state_dict()}
        restored = load_context_maskgit_v4(checkpoint).eval()
        model.eval()
        with torch.no_grad():
            self.assertTrue(torch.equal(model(ids, labels), restored(ids, labels)))


if __name__ == "__main__":
    unittest.main()
