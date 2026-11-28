import unittest
import torch
from models import TaiChi
from diagnose_stage2 import image_variants, qa_score, first_per_class


class DiagnosticControls(unittest.TestCase):
    def test_replacements_preserve_targets_and_cross_class(self):
        records=[(torch.full((3,32,32),float(c)), 'question',str(c),c) for c in [0,0,1,1,2,2]]
        v=image_variants(records)
        self.assertEqual(first_per_class(records),[0,2,4])
        for original,wrong,zero in zip(records,v['wrong_class'],v['zero']):
            self.assertEqual(original[1:],wrong[1:]); self.assertEqual(original[1:],zero[1:])
            self.assertFalse(torch.equal(original[0],wrong[0]))
            self.assertEqual(float(zero[0].abs().sum()),0.)

    def test_matched_noise_and_global_rng_preserved(self):
        torch.set_num_threads(2); torch.manual_seed(67)
        model=TaiChi().eval()
        records=[(torch.rand(3,32,32),'Object?','tench',0)]
        state=torch.get_rng_state().clone()
        one,l1=qa_score(model,records,'awgn',10.,80)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        two,l2=qa_score(model,records,'awgn',10.,80)
        self.assertEqual(one,two); torch.testing.assert_close(l1,l2,rtol=0,atol=0)


if __name__=='__main__': unittest.main()
