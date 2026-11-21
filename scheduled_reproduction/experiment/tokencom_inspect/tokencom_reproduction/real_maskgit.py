"""Trainable compact MaskGIT matched to the public 1024-entry VQGAN codebook."""
import torch
from torch.nn import functional as F
from models import MaskGIT,transformer


class RealMaskGIT(MaskGIT):
    def __init__(self,classes=10,d=128,layers=4):
        super().__init__(codes=1024,tokens=256,classes=classes,d=d)
        self.net=transformer(d,layers)
        self.config={'classes':classes,'d':d,'layers':layers}

    def training_loss(self,ids,labels):
        ratio=torch.cos(torch.rand(ids.shape[0],1,device=ids.device)*torch.pi/2)
        mask=torch.rand(ids.shape,device=ids.device)<ratio
        empty=~mask.any(1)
        if empty.any():
            rows=empty.nonzero().flatten()
            mask[rows,torch.randint(ids.shape[1],(len(rows),),device=ids.device)]=True
        condition=labels.masked_fill(torch.rand(labels.shape,device=ids.device)<.15,self.classes)
        return F.cross_entropy(self(ids.masked_fill(mask,self.codes),condition)[mask],ids[mask])


def load_real_maskgit(checkpoint,device='cpu'):
    if checkpoint['model']!='maskgit_real': raise ValueError('Expected real-data MaskGIT checkpoint')
    m=RealMaskGIT(**checkpoint['config']).to(device)
    m.load_state_dict(checkpoint['state'])
    return m
