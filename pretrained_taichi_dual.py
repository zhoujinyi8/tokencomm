"""TaiChi bridge for two distinct frozen visual backbones: ViT-B/16 and ConvNeXt-Tiny."""
import torch
from torch import nn

from models import TaiChi


class DualBackboneTaiChi(TaiChi):
    def __init__(self):
        super().__init__()
        for name in ['lr_patch', 'lr_pos', 'lr_vit', 'hr_cnn']:
            delattr(self, name)
        self.vit_adapter = nn.Sequential(nn.Conv2d(768, 64, 1), nn.GroupNorm(1, 64))
        self.conv_adapter = nn.Sequential(nn.Conv2d(768, 64, 1), nn.GroupNorm(1, 64))

    def encode(self, image, question, snr_db=10., kind='awgn'):
        vit, conv = image
        if vit.ndim != 4 or vit.shape[1:] != (768, 4, 4):
            raise ValueError('Expected cached ViT-B/16 features shaped [B,768,4,4]')
        if conv.ndim != 4 or conv.shape[1:] != (768, 7, 7):
            raise ValueError('Expected cached ConvNeXt-Tiny features shaped [B,768,7,7]')
        b = vit.shape[0]
        visual = self.kan(self.ban(self.vit_adapter(vit), self.conv_adapter(conv)).flatten(2).transpose(1, 2))
        x = torch.cat([visual, self.embedding(question)], 1)
        pad = torch.cat([torch.zeros(b, 16, dtype=torch.bool, device=vit.device), question.eq(0)], 1)
        x = x + self.position(torch.arange(x.shape[1], device=x.device))
        z = self.encoder(x, src_key_padding_mask=pad)
        if kind != 'none':
            out = torch.zeros_like(z)
            for i in range(b):
                valid = ~pad[i]
                y = self.channel_encoder(z[i:i+1, valid])
                out[i:i+1, valid] = self.channel_decoder(self.channel(y, snr_db, kind))
            z = out
        return z, pad

    def initialize_shared(self, checkpoint):
        state = {k: v for k, v in checkpoint['state'].items()
                 if not k.startswith(('lr_adapter.', 'hr_adapter.', 'vit_adapter.', 'conv_adapter.'))}
        result = self.load_state_dict(state, strict=False)
        if result.unexpected_keys:
            raise ValueError(f'Unexpected warm-start keys: {result.unexpected_keys}')
        if not result.missing_keys or not all(k.startswith(('vit_adapter.', 'conv_adapter.')) for k in result.missing_keys):
            raise ValueError(f'Unexpected missing warm-start keys: {result.missing_keys}')


def load_dual_taichi(checkpoint):
    if checkpoint['model'] != 'taichi_vit_b16_convnext_tiny_features':
        raise ValueError('Wrong model checkpoint')
    model = DualBackboneTaiChi()
    model.load_state_dict(checkpoint['state'], strict=True)
    return model
