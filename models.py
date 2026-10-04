"""Small, independently implemented research baselines; not author checkpoints."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def transformer(d=64, layers=2):
    return nn.TransformerEncoder(nn.TransformerEncoderLayer(
        d, 4, d * 4, dropout=0., batch_first=True, norm_first=True), layers,
        enable_nested_tensor=False)


class SplineKAN(nn.Module):
    """Edge-wise SiLU + cubic B-spline functions, fixed uniform knots."""
    def __init__(self, din, dout, grid=8, degree=3):
        super().__init__()
        self.degree = degree
        self.register_buffer('knots', torch.arange(-degree, grid+degree+1)*2./grid-1.)
        self.base = nn.Linear(din, dout, bias=False)
        self.coeff = nn.Parameter(torch.randn(dout, din, grid+degree)*.02)

    def basis(self, x):
        x = x.unsqueeze(-1)
        t = self.knots
        b = ((x >= t[:-1]) & (x < t[1:])).to(x.dtype)
        for k in range(1, self.degree+1):
            b = ((x-t[:-(k+1)])/(t[k:-1]-t[:-(k+1)])*b[..., :-1]
                 +(t[k+1:]-x)/(t[k+1:]-t[1:-k])*b[..., 1:])
        return b

    def forward(self, x):
        x = torch.tanh(x)
        return self.base(F.silu(x)) + torch.einsum('...ig,oig->...o', self.basis(x), self.coeff)


class LocalAttention(nn.Module):
    """Queries attend to a spatially aligned 3x3 neighborhood; no dense HW x HW map."""
    def __init__(self, d, window=3):
        super().__init__()
        self.window = window
        self.q, self.k, self.v = (nn.Conv2d(d, d, 1) for _ in range(3))
        self.out = nn.Conv2d(d, d, 1)

    def forward(self, q, context):
        b, d, h, w = q.shape
        context = F.interpolate(context, (h, w), mode='bilinear', align_corners=False)
        n = self.window ** 2
        k = F.unfold(self.k(context), self.window, padding=self.window//2).view(b,d,n,h*w)
        v = F.unfold(self.v(context), self.window, padding=self.window//2).view(b,d,n,h*w)
        score = (self.q(q).flatten(2).unsqueeze(2)*k).sum(1)/math.sqrt(d)
        valid = F.unfold(q.new_ones(1,1,h,w), self.window, padding=self.window//2).bool()
        score = score.masked_fill(~valid, -torch.inf)
        return self.out((score.softmax(1).unsqueeze(1)*v).sum(2).view(b,d,h,w))


class BAN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lr_reads_hr = LocalAttention(d)
        self.hr_reads_lr = LocalAttention(d)

    def forward(self, lr, hr):
        return lr + self.lr_reads_hr(lr, hr) + F.adaptive_avg_pool2d(
            self.hr_reads_lr(hr, lr), lr.shape[-2:])


class Channel(nn.Module):
    """Complex symbols, Es/N0 dB, unit mean complex-symbol power, perfect receiver CSI."""
    def forward(self, x, snr_db=10., kind='awgn'):
        if kind == 'none':
            return x
        if kind not in ('awgn', 'rayleigh') or x.shape[-1] % 2:
            raise ValueError('Expected awgn/rayleigh and even channel width')
        z = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2).contiguous())
        z = z / z.abs().square().mean((-2,-1), keepdim=True).clamp_min(1e-8).sqrt()
        n0 = 10.**(-float(snr_db)/10.)
        noise = torch.complex(torch.randn_like(z.real),torch.randn_like(z.real))*math.sqrt(n0/2)
        if kind == 'rayleigh':
            h = torch.complex(torch.randn_like(z.real),torch.randn_like(z.real))/math.sqrt(2)
            # MMSE equalization; receiver knows h. No outage/retransmission model.
            z = (h*z+noise)*h.conj()/(h.abs().square()+n0)
        else:
            z = z+noise
        return torch.view_as_real(z).flatten(-2)


class TaiChi(nn.Module):
    """Tiny dual-vision/BAN/KAN + encoder/channel/autoregressive decoder surrogate."""
    def __init__(self, vocab=260, d=64):
        super().__init__()
        self.lr_patch = nn.Conv2d(3,d,4,4)
        self.lr_pos = nn.Parameter(torch.randn(1,16,d)*.02)
        self.lr_vit = transformer(d,1)
        self.hr_cnn = nn.Sequential(nn.Conv2d(3,d//2,3,2,1),nn.GELU(),nn.Conv2d(d//2,d,3,2,1))
        self.ban = BAN(d)
        self.kan = nn.Sequential(nn.LayerNorm(d),SplineKAN(d,d),SplineKAN(d,d))
        self.embedding = nn.Embedding(vocab,d,padding_idx=0)
        self.position = nn.Embedding(512,d)
        self.encoder = transformer(d)
        self.channel_encoder = nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,16))
        self.channel = Channel()
        self.channel_decoder = nn.Sequential(nn.Linear(16,d),nn.GELU(),nn.Linear(d,d))
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(
            d,4,d*4,dropout=0.,batch_first=True,norm_first=True),2)
        self.head = nn.Linear(d,vocab)

    def encode(self, image, question, snr_db=10., kind='awgn'):
        b = image.shape[0]
        lr = self.lr_patch(F.interpolate(image,(16,16),mode='bilinear',align_corners=False))
        lr = self.lr_vit(lr.flatten(2).transpose(1,2)+self.lr_pos).transpose(1,2).reshape(b,-1,4,4)
        visual = self.kan(self.ban(lr,self.hr_cnn(image)).flatten(2).transpose(1,2))
        x = torch.cat([visual,self.embedding(question)],1)
        pad = torch.cat([torch.zeros(b,16,dtype=torch.bool,device=image.device),question.eq(0)],1)
        x = x+self.position(torch.arange(x.shape[1],device=x.device))
        z = self.encoder(x,src_key_padding_mask=pad)
        if kind != 'none':
            # Avoid spending transmit power on padding: process each valid sequence separately.
            out = torch.zeros_like(z)
            for i in range(b):
                valid = ~pad[i]
                y = self.channel_encoder(z[i:i+1,valid])
                out[i:i+1,valid] = self.channel_decoder(self.channel(y,snr_db,kind))
            z = out
        return z,pad

    def decode(self, memory, pad, ids):
        x = self.embedding(ids)+self.position(torch.arange(ids.shape[1],device=ids.device))
        mask = torch.ones(ids.shape[1],ids.shape[1],device=ids.device,dtype=torch.bool).triu(1)
        h = self.decoder(x,memory,tgt_mask=mask,tgt_key_padding_mask=ids.eq(0),memory_key_padding_mask=pad)
        return self.head(h)

    def forward(self, image, question, answer_input, snr_db=10., kind='awgn'):
        return self.decode(*self.encode(image,question,snr_db,kind),answer_input)

    @torch.no_grad()
    def generate(self, image, question, snr_db=10., kind='awgn', max_new_tokens=40):
        memory,pad = self.encode(image,question,snr_db,kind)
        ids = torch.ones(memory.shape[0],1,dtype=torch.long,device=memory.device)
        done = torch.zeros(memory.shape[0],dtype=torch.bool,device=memory.device)
        for _ in range(max_new_tokens):
            nxt = self.decode(memory,pad,ids)[:,-1].argmax(-1)
            nxt = torch.where(done,torch.zeros_like(nxt),nxt)
            ids = torch.cat([ids,nxt[:,None]],1)
            done |= nxt.eq(2)
            if done.all(): break
        return ids

    def set_stage(self, stage):
        if stage not in ('scratch','align','sft','joint'): raise ValueError(stage)
        for p in self.parameters(): p.requires_grad_(stage=='scratch')
        modules = {'align':[self.ban,self.kan],
                   'sft':[self.ban,self.kan,self.embedding,self.position,self.encoder,self.decoder,self.head],
                   'joint':[self.ban,self.kan,self.embedding,self.position,self.encoder,self.decoder,self.head,
                            self.channel_encoder,self.channel_decoder]}
        for m in modules.get(stage,[]):
            for p in m.parameters(): p.requires_grad_(True)


class VQTokenizer(nn.Module):
    """VQ-VAE smoke backend. Does NOT implement VQGAN adversarial/perceptual training."""
    def __init__(self, codes=64, d=32):
        super().__init__()
        self.codes = codes
        self.encoder = nn.Sequential(nn.Conv2d(3,32,4,2,1),nn.GELU(),nn.Conv2d(32,d,4,2,1))
        self.codebook = nn.Embedding(codes,d)
        nn.init.uniform_(self.codebook.weight,-1/codes,1/codes)
        self.decoder = nn.Sequential(nn.ConvTranspose2d(d,32,4,2,1),nn.GELU(),nn.ConvTranspose2d(32,3,4,2,1),nn.Sigmoid())

    def quantize(self, image):
        z = self.encoder(image).permute(0,2,3,1)
        flat = z.reshape(-1,z.shape[-1]); w = self.codebook.weight
        ids = (flat.square().sum(1,keepdim=True)+w.square().sum(1)-2*flat@w.T).argmin(1)
        q = self.codebook(ids).view_as(z)
        loss = F.mse_loss(q,z.detach())+.25*F.mse_loss(z,q.detach())
        return z, q, ids.view(image.shape[0],-1), loss

    def forward(self, image):
        z,q,ids,vq = self.quantize(image)
        recon = self.decoder((z+(q-z).detach()).permute(0,3,1,2))
        return recon,ids,F.mse_loss(recon,image)+vq

    @torch.no_grad()
    def encode(self,image): return self.quantize(image)[2]

    def decode(self,ids):
        s = math.isqrt(ids.shape[1])
        if s*s != ids.shape[1]: raise ValueError('Token count must be square')
        return self.decoder(self.codebook(ids).view(ids.shape[0],s,s,-1).permute(0,3,1,2))


def erase_packets(ids, per, packet_tokens=4):
    """Shared random interleaver; perfect error detection; PER is input, not derived SNR."""
    if not 0 <= per <= 1 or packet_tokens < 1: raise ValueError('Invalid packet parameters')
    b,n = ids.shape
    order = torch.randperm(n,device=ids.device)
    lost = torch.rand(b,math.ceil(n/packet_tokens),device=ids.device)<per
    mask = torch.zeros_like(ids,dtype=torch.bool)
    mask[:,order] = lost.repeat_interleave(packet_tokens,1)[:,:n]
    return mask


class MaskGIT(nn.Module):
    def __init__(self, codes=64, tokens=64, classes=3, d=64):
        super().__init__()
        self.codes, self.classes = codes, classes
        self.embedding = nn.Embedding(codes+1,d)
        self.condition = nn.Embedding(classes+1,d)  # Last ID means no CMI.
        self.position = nn.Parameter(torch.randn(1,tokens,d)*.02)
        self.net = transformer(d)
        self.head = nn.Linear(d,codes)

    def forward(self, ids, labels):
        x = self.embedding(ids)+self.position+self.condition(labels)[:,None]
        return self.head(self.net(x))

    def masked_loss(self, ids, labels):
        ratio = torch.cos(torch.rand(ids.shape[0],1,device=ids.device)*math.pi/2)
        mask = torch.rand_like(ids,dtype=torch.float)<ratio
        mask[:,0] = True
        return F.cross_entropy(self(ids.masked_fill(mask,self.codes),labels)[mask],ids[mask])

    @torch.no_grad()
    def recover(self, ids, missing, labels, steps=8):
        if steps < 1: raise ValueError('steps must be positive')
        x = ids.masked_fill(missing,self.codes).clone()
        remaining = missing.clone(); initial = missing.sum(1)
        for step in range(steps):
            probs = self(x,labels).softmax(-1)
            confidence,pred = probs.max(-1)
            for b in range(x.shape[0]):
                idx = remaining[b].nonzero().flatten()
                target = int(initial[b]*math.cos((step+1)/steps*math.pi/2))
                take = idx.numel()-target
                if take > 0:
                    chosen = idx[confidence[b,idx].topk(take).indices]
                    x[b,chosen] = pred[b,chosen]; remaining[b,chosen] = False
        return x
