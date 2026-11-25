import unittest
import torch
from torch.nn import functional as F
from models import TaiChi, SplineKAN, Channel, VQTokenizer, MaskGIT, erase_packets


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2); torch.manual_seed(7)

    def test_spline_partition_and_gradient(self):
        m=SplineKAN(3,4); x=torch.linspace(-.99,.99,30).view(10,3).requires_grad_()
        self.assertTrue(torch.allclose(m.basis(x).sum(-1),torch.ones_like(x),atol=1e-5))
        m(x).square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(m.coeff.grad.abs().sum().item(),0)

    def test_channel_power_and_noise(self):
        x=torch.randn(16,1000,16); c=Channel()
        z=x.view(16,1000,8,2)
        norm=(z.square().sum(-1).mean((1,2),keepdim=True).sqrt().unsqueeze(-1))
        clean=(z/norm).flatten(-2)
        low=(c(x,0)-clean).square().mean(); high=(c(x,20)-clean).square().mean()
        self.assertGreater(float(low/high),80); self.assertLess(float(low/high),120)
        self.assertTrue(torch.isfinite(c(x,-5,'rayleigh')).all())

    def test_taichi_backward_causal_and_padding(self):
        m=TaiChi().eval(); image=torch.rand(2,3,32,32); q=torch.tensor([[4,5,0],[6,7,8]])
        a=torch.tensor([[1,10,11],[1,12,13]])
        logits=m(image,q,a,kind='awgn'); loss=F.cross_entropy(logits.flatten(0,1),a.flatten())
        loss.backward()
        for name in ['ban','kan','channel_encoder','channel_decoder','encoder','decoder']:
            grads=[p.grad for p in getattr(m,name).parameters() if p.grad is not None]
            self.assertTrue(grads,name); self.assertTrue(all(torch.isfinite(g).all() for g in grads),name)
            self.assertGreater(sum(g.abs().sum().item() for g in grads),0,name)
        mem,pad=m.encode(image,q,kind='none')
        a2=a.clone(); a2[:,2]=99
        self.assertTrue(torch.allclose(m.decode(mem,pad,a)[:,:2],m.decode(mem,pad,a2)[:,:2],atol=1e-6))
        q2=F.pad(q,(0,2)); mem2,pad2=m.encode(image,q2,kind='none')
        self.assertTrue(torch.allclose(m.decode(mem,pad,a),m.decode(mem2,pad2,a),atol=1e-5))

    def test_packet_extremes_and_recovery_preservation(self):
        ids=torch.randint(0,64,(2,64)); model=MaskGIT().eval(); labels=torch.tensor([0,1])
        self.assertFalse(erase_packets(ids,0).any()); self.assertTrue(erase_packets(ids,1).all())
        for per in [0.,.5,1.]:
            mask=erase_packets(ids,per); result=model.recover(ids,mask,labels)
            self.assertTrue(torch.equal(result[~mask],ids[~mask]))
            self.assertTrue(((result>=0)&(result<64)).all())

    def test_vq_updates_and_shapes(self):
        m=VQTokenizer(); image=torch.rand(2,3,32,32)
        rec,ids,loss=m(image); loss.backward()
        self.assertEqual(rec.shape,image.shape); self.assertEqual(ids.shape,(2,64))
        self.assertGreater(m.codebook.weight.grad.abs().sum().item(),0)
        self.assertEqual(m.decode(ids).shape,image.shape)

    def test_freezing(self):
        m=TaiChi(); m.set_stage('align')
        for name,p in m.named_parameters(): self.assertEqual(p.requires_grad,name.startswith(('ban.','kan.')))


if __name__=='__main__': unittest.main(verbosity=2)
