"""A no-CMI 2-D local-context MaskGIT receiver for a future, unused data split."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from models import transformer


class LocalBlock(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.norm=nn.GroupNorm(8,d)
        self.depthwise=nn.Conv2d(d,d,3,padding=1,groups=d)
        self.pointwise=nn.Conv2d(d,d,1)
        self.gate=nn.Sequential(nn.Conv2d(d,2*d,1),nn.GLU(1))

    def forward(self,x):
        y=self.pointwise(F.gelu(self.depthwise(self.norm(x))))
        return x+y+self.gate(self.norm(x))


class ContextMaskGITv2(nn.Module):
    """No class side information; combines 2-D local and global token context."""
    codes=1024;tokens=256
    def __init__(self,d=192,layers=6,local_blocks=3):
        super().__init__()
        if d%8:raise ValueError('d must divide eight groups')
        self.d=d;self.config={'d':d,'layers':layers,'local_blocks':local_blocks,'condition':'none'}
        self.embedding=nn.Embedding(self.codes+1,d)
        self.row=nn.Parameter(torch.randn(1,16,1,d)*.02);self.column=nn.Parameter(torch.randn(1,1,16,d)*.02)
        self.local=nn.Sequential(*[LocalBlock(d) for _ in range(local_blocks)])
        self.global_net=transformer(d,layers);self.norm=nn.LayerNorm(d);self.head=nn.Linear(d,self.codes)

    def forward(self,ids):
        if ids.ndim!=2 or ids.shape[1]!=256:raise ValueError('Expected [batch,256] token IDs')
        x=self.embedding(ids).view(-1,16,16,self.d)+self.row+self.column
        x=self.local(x.permute(0,3,1,2)).permute(0,2,3,1).reshape(-1,256,self.d)
        return self.head(self.norm(self.global_net(x)))

    @torch.no_grad()
    def recover(self,ids,missing,steps=8):
        if steps<1:raise ValueError('steps must be positive')
        x=ids.masked_fill(missing,self.codes).clone();remaining=missing.clone();initial=missing.sum(1)
        for step in range(steps):
            confidence,pred=self(x).softmax(-1).max(-1)
            for b in range(len(x)):
                idx=remaining[b].nonzero().flatten();target=int(initial[b]*math.cos((step+1)/steps*math.pi/2));take=len(idx)-target
                if take>0:
                    chosen=idx[confidence[b,idx].topk(take).indices];x[b,chosen]=pred[b,chosen];remaining[b,chosen]=False
        return x

    def loss(self,ids,packet_tokens=4,per=.3):
        from models import erase_packets
        missing=erase_packets(ids,per,packet_tokens)
        return F.cross_entropy(self(ids.masked_fill(missing,self.codes))[missing],ids[missing]),missing


def load_context_maskgit_v2(checkpoint,device='cpu'):
    if checkpoint['model']!='context_maskgit_v2':raise ValueError('Expected ContextMaskGITv2 checkpoint')
    config={k:v for k,v in checkpoint['config'].items() if k in {'d','layers','local_blocks'}}
    model=ContextMaskGITv2(**config).to(device);model.load_state_dict(checkpoint['state']);return model
