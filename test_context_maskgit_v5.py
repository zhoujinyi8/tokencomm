import unittest

import torch

from context_maskgit_v4 import ContextMaskGITv4
from context_maskgit_v5 import initialize_context_maskgit_v5_from_v4, load_context_maskgit_v5


class ContextMaskGITv5Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20)
        self.v4 = ContextMaskGITv4(d=16, layers=1, local_blocks=1, classes=10).eval()
        self.checkpoint = {"model": "context_maskgit_v4", "config": self.v4.config, "state": self.v4.state_dict()}
        self.text = torch.randn(3, 8)
        self.names = ["alpha", "beta", "gamma"]

    def test_unconditional_warm_start_is_exact(self):
        v5 = initialize_context_maskgit_v5_from_v4(self.checkpoint, self.text, self.names).eval()
        ids = torch.randint(0, 1024, (2, 256))
        self.assertTrue(torch.equal(self.v4(ids, None), v5(ids, None)))
        for label in range(3):
            labels = torch.full((2,), label)
            self.assertTrue(torch.equal(v5(ids, labels), v5(ids, None)))

    def test_text_projection_receives_gradient(self):
        v5 = initialize_context_maskgit_v5_from_v4(self.checkpoint, self.text, self.names).train()
        ids = torch.randint(0, 1024, (2, 256))
        labels = torch.tensor([0, 1])
        v5(ids, labels).square().mean().backward()
        gradient = v5.condition.projection.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_checkpoint_round_trip(self):
        model = initialize_context_maskgit_v5_from_v4(self.checkpoint, self.text, self.names).eval()
        checkpoint = {"model": "context_maskgit_v5", "config": model.config, "state": model.state_dict()}
        loaded = load_context_maskgit_v5(checkpoint).eval()
        ids = torch.randint(0, 1024, (1, 256))
        labels = torch.tensor([2])
        self.assertTrue(torch.equal(model(ids, labels), loaded(ids, labels)))


if __name__ == "__main__":
    unittest.main()
