import io
import unittest
import torch
from real_maskgit import RealMaskGIT,load_real_maskgit


class RealTrainingTests(unittest.TestCase):
    def test_optimizer_and_rng_resume_equivalence(self):
        torch.set_num_threads(2); torch.manual_seed(87)
        model=RealMaskGIT(classes=10,d=32,layers=1)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        ids=torch.randint(0,1024,(2,256)); labels=torch.tensor([0,1])
        def step(m,opt):
            m.train(); opt.zero_grad(); loss=m.training_loss(ids,labels)
            loss.backward(); opt.step(); return loss.detach()
        step(model,optimizer)
        buf=io.BytesIO()
        torch.save({'model':'maskgit_real','config':model.config,'state':model.state_dict(),
                    'optimizer':optimizer.state_dict(),'rng':torch.get_rng_state()},buf)
        expected_loss=step(model,optimizer)
        buf.seek(0); checkpoint=torch.load(buf,weights_only=True)
        restored=load_real_maskgit(checkpoint)
        opt2=torch.optim.AdamW(restored.parameters(),lr=.001); opt2.load_state_dict(checkpoint['optimizer'])
        torch.set_rng_state(checkpoint['rng'])
        actual_loss=step(restored,opt2)
        torch.testing.assert_close(actual_loss,expected_loss,rtol=0,atol=0)
        for name,value in model.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name],value,rtol=0,atol=0)


if __name__=='__main__': unittest.main(verbosity=2)
