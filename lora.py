"""LoRA on real projection weights via parametrizations, including fused MHA weights."""
import math
import torch
from torch import nn
from torch.nn.utils import parametrize
from models import TaiChi


class LowRankDelta(nn.Module):
    def __init__(self,weight,rank=4,alpha=8.):
        super().__init__()
        if rank<1 or rank>min(weight.shape): raise ValueError('Invalid rank')
        self.scale=float(alpha)/rank
        self.A=nn.Parameter(weight.new_empty(rank,weight.shape[1]))
        self.B=nn.Parameter(weight.new_zeros(weight.shape[0],rank))
        nn.init.kaiming_uniform_(self.A,a=math.sqrt(5))

    def forward(self,weight): return weight+self.scale*(self.B@self.A)


def attach_lora(model,rank=4,alpha=8.):
    targets=[]
    # Snapshot before parametrization adds children. Only the language encoder/decoder.
    for name,module in list(model.named_modules()):
        if not name.startswith(('encoder.','decoder.')): continue
        attr='in_proj_weight' if isinstance(module,nn.MultiheadAttention) else 'weight' if isinstance(module,nn.Linear) else None
        if attr:
            if parametrize.is_parametrized(module,attr): raise ValueError('LoRA already attached')
            weight=getattr(module,attr)
            parametrize.register_parametrization(module,attr,LowRankDelta(weight,rank,alpha))
            targets.append(name+'.'+attr)
    return targets


def configure_joint_lora(model):
    for name,p in model.named_parameters():
        p.requires_grad_(name.startswith(('ban.','kan.','channel_encoder.','channel_decoder.'))
                         or '.parametrizations.' in name and name.endswith(('.A','.B')))


def load_taichi(checkpoint,device='cpu'):
    if checkpoint['model']!='taichi': raise ValueError('Expected TaiChi checkpoint')
    model=TaiChi().to(device)
    if checkpoint.get('lora'):
        attach_lora(model,checkpoint['lora']['rank'],checkpoint['lora']['alpha'])
    model.load_state_dict(checkpoint['state'])
    return model
