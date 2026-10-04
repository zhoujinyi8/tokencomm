import unittest
import torch
from models import TaiChi
from pretrained_taichi import FeatureTaiChi,load_feature_taichi
from train_pretrained_taichi import get_batch


class FeatureBridgeTests(unittest.TestCase):
    def test_adapter_gradients_and_reload(self):
        torch.set_num_threads(2);torch.manual_seed(72)
        model=FeatureTaiChi();model.initialize_language_and_fusion({'state':TaiChi().state_dict()})
        im=(torch.randn(2,512,4,4),torch.randn(2,256,8,8));q=torch.tensor([[7,8],[9,0]]);a=torch.tensor([[1,10],[1,11]])
        out=model(im,q,a,120.,'awgn');self.assertEqual(out.shape,(2,2,260))
        out.square().mean().backward()
        for name in ['lr_adapter','hr_adapter','ban','kan','channel_encoder','channel_decoder']:
            g=[p.grad for p in getattr(model,name).parameters()]
            self.assertTrue(all(x is not None and torch.isfinite(x).all() for x in g),name)
            self.assertGreater(sum(float(x.abs().sum()) for x in g),0.,name)
        model.eval();loaded=load_feature_taichi({'model':'taichi_resnet18_features','state':model.state_dict()}).eval()
        torch.manual_seed(20);expected=model(im,q,a,10.,'awgn')
        torch.manual_seed(20);actual=loaded(im,q,a,10.,'awgn')
        torch.testing.assert_close(expected,actual)
        torch.manual_seed(20);expected_ids=model.generate(im,q,10.,'awgn',max_new_tokens=3)
        torch.manual_seed(20);actual_ids=loaded.generate(im,q,10.,'awgn',max_new_tokens=3)
        self.assertEqual(expected_ids.shape[0],2)
        self.assertTrue(torch.equal(expected_ids,actual_ids))

    def test_feature_ablation_keeps_question_and_answer(self):
        labels=torch.arange(10)
        data={'labels':labels,'lr':torch.arange(10.).view(10,1,1,1).expand(10,512,4,4),
              'hr':torch.arange(10.).view(10,1,1,1).expand(10,256,8,8),
              'records':[{'question':'object?','answer':str(i)} for i in range(10)]}
        correct,q,a=get_batch(data,[1,2]);wrong,q2,a2=get_batch(data,[1,2],'wrong_class');zero,_,_=get_batch(data,[1,2],'zero')
        self.assertTrue(torch.equal(q,q2));self.assertTrue(torch.equal(a,a2))
        self.assertTrue(torch.equal(wrong[0][:,0,0,0],torch.tensor([2.,3.])))
        self.assertEqual(float(zero[0].sum()+zero[1].sum()),0.)


if __name__=='__main__':unittest.main()
