import unittest
import torch
from context_maskgit_v2 import ContextMaskGITv2,load_context_maskgit_v2


class ContextV2Tests(unittest.TestCase):
    def test_shapes_gradients_and_recovery_preserves_received_tokens(self):
        torch.set_num_threads(2);torch.manual_seed(91)
        model=ContextMaskGITv2(d=32,layers=1,local_blocks=1)
        ids=torch.randint(0,1024,(2,256));loss,missing=model.loss(ids)
        self.assertTrue(torch.isfinite(loss));loss.backward()
        self.assertGreater(float(model.row.grad.abs().sum()),0.)
        self.assertGreater(float(model.local[0].depthwise.weight.grad.abs().sum()),0.)
        recovered=model.recover(ids.masked_fill(missing,1024),missing,steps=3)
        self.assertTrue(torch.equal(recovered[~missing],ids[~missing]))
        ck={'model':'context_maskgit_v2','config':model.config,'state':model.state_dict()}
        loaded=load_context_maskgit_v2(ck);self.assertEqual(loaded(ids).shape,(2,256,1024))


if __name__=='__main__':unittest.main()
