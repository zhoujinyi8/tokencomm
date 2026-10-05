import unittest
import torch

from pretrained_taichi_dual import DualBackboneTaiChi, load_dual_taichi


class DualBackboneTaiChiTests(unittest.TestCase):
    def test_shapes_gradients_generation_and_reload(self):
        torch.set_num_threads(2); torch.manual_seed(8810)
        model=DualBackboneTaiChi()
        images=(torch.randn(2,768,4,4),torch.randn(2,768,7,7)); q=torch.randint(3,40,(2,5)); ans=torch.randint(3,40,(2,4))
        logits=model(images,q,ans,10.,'awgn'); self.assertEqual(logits.shape,(2,4,260))
        logits.mean().backward(); self.assertGreater(float(model.vit_adapter[0].weight.grad.abs().sum()),0.)
        self.assertGreater(float(model.conv_adapter[0].weight.grad.abs().sum()),0.)
        ids=model.generate(images,q,120.,'awgn',max_new_tokens=2); self.assertEqual(ids.shape[0],2)
        ck={'model':'taichi_vit_b16_convnext_tiny_features','state':model.state_dict()}
        loaded=load_dual_taichi(ck); self.assertEqual(loaded(images,q,ans,10.,'none').shape,(2,4,260))


if __name__=='__main__': unittest.main()
