import unittest
import torch
from models import TaiChi
from lora import attach_lora,configure_joint_lora,load_taichi


class LoRATests(unittest.TestCase):
    def test_initial_equivalence_frozen_base_updates_and_reload(self):
        torch.set_num_threads(2); torch.manual_seed(44)
        m=TaiChi().eval(); image=torch.rand(2,3,32,32)
        q=torch.tensor([[6,7],[8,9]]); answer=torch.tensor([[1,12],[1,14]])
        expected=m(image,q,answer,kind='none').detach()
        attach_lora(m,4,8.)
        torch.testing.assert_close(m(image,q,answer,kind='none'),expected,atol=1e-6,rtol=1e-5)
        configure_joint_lora(m)
        before={n:p.detach().clone() for n,p in m.named_parameters() if not p.requires_grad}
        m.train(); opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.01)
        m(image,q,answer,kind='rayleigh').square().mean().backward()
        grads=[p.grad for n,p in m.named_parameters() if n.endswith('.B')]
        self.assertTrue(grads); self.assertTrue(all(g is not None and torch.isfinite(g).all() and g.abs().sum()>0 for g in grads))
        opt.step()
        for n,p in m.named_parameters():
            if n in before: self.assertTrue(torch.equal(p,before[n]),n)
        m.eval(); ck={'model':'taichi','state':m.state_dict(),'lora':{'rank':4,'alpha':8.}}
        loaded=load_taichi(ck).eval()
        torch.testing.assert_close(loaded(image,q,answer,kind='none'),m(image,q,answer,kind='none'))


if __name__=='__main__': unittest.main(verbosity=2)
