import unittest
from pathlib import Path
import torch

from qwen_dual_taichi import QwenDualTaiChi


LLM=Path(r'D:\新建文件夹 (2)\2026-09-21-14-31-36\taichi-tokecom\models\qwen2.5-0.5b-instruct')


@unittest.skipUnless(LLM.exists(),'local public Qwen checkpoint unavailable')
class QwenDualTaiChiTests(unittest.TestCase):
    def test_forward_trainable_scope_and_generation(self):
        torch.set_num_threads(2); torch.manual_seed(9910)
        model=QwenDualTaiChi(LLM,lora_rank=2)
        stats=model.trainable_parameter_summary(); self.assertLess(stats['trainable_fraction'],.05); self.assertGreater(stats['trainable_parameters'],0)
        images=(torch.randn(1,768,4,4),torch.randn(1,768,7,7))
        out=model(images,['What is the main object in this image?'],['tench'],120.,'none'); self.assertTrue(torch.isfinite(out.loss)); out.loss.backward()
        self.assertGreater(float(model.vit_adapter[0].weight.grad.abs().sum()),0.)
        answer=model.generate_answers(images,['What is the main object in this image?'],120.,'none',max_new_tokens=2)
        self.assertEqual(len(answer),1)


if __name__=='__main__': unittest.main()
