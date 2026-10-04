"""Strictly matched public MaskGIT VQGAN, 1024 entries, 16x16 tokens.

The architecture is vendored unchanged with attribution in third_party/.
The wrapper uses SimpleNamespace to avoid an unnecessary OmegaConf dependency.
"""
import hashlib
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from third_party.maskgit_vqgan import Encoder,Decoder,VectorQuantizer

WEIGHT_SHA256='424e7f2b11d1d63691ae3fbab74cf6c417ecbe0d8f86187b71f3538c7b56f6d2'
DEFAULT_WEIGHT=Path(__file__).parent/'pretrained/maskgit-vqgan-imagenet-f16-256.bin'


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''): h.update(part)
    return h.hexdigest()


class PretrainedVQGAN(nn.Module):
    def __init__(self,weights=DEFAULT_WEIGHT):
        super().__init__()
        if sha256(weights)!=WEIGHT_SHA256: raise ValueError('Weight SHA256 does not match pinned public checkpoint')
        cfg=SimpleNamespace(channel_mult=[1,1,2,2,4],num_resolutions=5,dropout=0.,
                            hidden_channels=128,num_channels=3,num_res_blocks=2,resolution=256,z_channels=256)
        self.encoder=Encoder(cfg); self.decoder=Decoder(cfg)
        self.quantize=VectorQuantizer(num_embeddings=1024,embedding_dim=256,commitment_cost=.25)
        self.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True),strict=True)
        self.requires_grad_(False); self.eval()

    @torch.no_grad()
    def encode(self,image):
        if image.ndim!=4 or image.shape[1:]!=(3,256,256): raise ValueError('Expected [B,3,256,256] image in [0,1]')
        if not torch.isfinite(image).all() or image.min()<0 or image.max()>1: raise ValueError('Image must be finite and in [0,1]')
        _,ids,_=self.quantize(self.encoder(image))
        return ids

    @torch.no_grad()
    def decode(self,ids):
        if ids.ndim!=2 or ids.shape[1]!=256 or ids.min()<0 or ids.max()>=1024: raise ValueError('Expected [B,256] codebook IDs')
        return self.decoder(self.quantize.get_codebook_entry(ids)).clamp(0,1)
