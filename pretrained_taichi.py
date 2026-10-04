"""Feature-adapter TaiChi: frozen shared ResNet18 features, not paper CLIP/LLM."""
import torch
from torch import nn
from models import TaiChi


class FeatureTaiChi(TaiChi):
    def __init__(self):
        super().__init__()
        for name in ['lr_patch','lr_pos','lr_vit','hr_cnn']:delattr(self,name)
        self.lr_adapter=nn.Sequential(nn.Conv2d(512,64,1),nn.GroupNorm(1,64))
        self.hr_adapter=nn.Sequential(nn.Conv2d(256,64,1),nn.GroupNorm(1,64))

    def encode(self,image,question,snr_db=10.,kind='awgn'):
        low,high=image
        if low.ndim!=4 or low.shape[1:]!=(512,4,4) or high.shape[1:]!=(256,8,8):
            raise ValueError('Expected cached ResNet18 low/high feature maps')
        b=low.shape[0]
        visual=self.kan(self.ban(self.lr_adapter(low),self.hr_adapter(high)).flatten(2).transpose(1,2))
        x=torch.cat([visual,self.embedding(question)],1)
        pad=torch.cat([torch.zeros(b,16,dtype=torch.bool,device=low.device),question.eq(0)],1)
        x=x+self.position(torch.arange(x.shape[1],device=x.device))
        z=self.encoder(x,src_key_padding_mask=pad)
        if kind!='none':
            out=torch.zeros_like(z)
            for i in range(b):
                valid=~pad[i]; y=self.channel_encoder(z[i:i+1,valid])
                out[i:i+1,valid]=self.channel_decoder(self.channel(y,snr_db,kind))
            z=out
        return z,pad

    def initialize_language_and_fusion(self,checkpoint):
        state={k:v for k,v in checkpoint['state'].items() if not k.startswith(('lr_patch.','lr_pos','lr_vit.','hr_cnn.'))}
        result=self.load_state_dict(state,strict=False)
        assert not result.unexpected_keys
        assert all(n.startswith(('lr_adapter.','hr_adapter.')) for n in result.missing_keys)


def load_feature_taichi(checkpoint):
    if checkpoint['model']!='taichi_resnet18_features':raise ValueError('Wrong model checkpoint')
    model=FeatureTaiChi();model.load_state_dict(checkpoint['state'],strict=True)
    return model
