#- *- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from Models.utils import repeat
from Models.layers import EncoderLayer#, ConformerEncoderLayer
from Models.modules import PositionalEncoder #RelativePositionalEncoder

class Encoder(nn.Module):
    def __init__(self, hp):
        super().__init__()
        d_model = hp.d_model_e
        self.N = hp.N_e
        self.heads = hp.heads
        xscale = math.sqrt(d_model)
        dropout = hp.dropout

        self.pe = PositionalEncoder(d_model, xscale=xscale, dropout=dropout)
        self.layers = repeat(self.N, lambda: EncoderLayer(d_model, self.heads, dropout))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, mask):
        x = self.pe(src)
        b, t, _ = x.shape
        attns_enc = torch.zeros((b, self.N, self.heads, t, t), device=x.device)
        for i in range(self.N):
            x, attn_enc = self.layers[i](x, mask)
            attns_enc[:,i] = attn_enc.detach()
        return self.norm(x), attns_enc

class ConformerEncoder(nn.Module):
    def __init__(self, d_model, N, heads, xscale, dropout):
        super().__init__()
        self.N = N
        self.heads = heads
        self.pe = RelativePositionalEncoder(d_model, xscale=1, dropout=dropout)
        self.layers = repeat(N, lambda: ConformerEncoderLayer(d_model, heads, dropout))

    def forward(self, src, mask):
        x, pe = self.pe(src)
        b, t, _ = x.shape
        attns_enc = torch.zeros((b, self.N, self.heads, t, t), device=x.device)
        for i in range(self.N):
            x, attn_enc = self.layers[i](x, pe, mask)
            attns_enc[:,i] = attn_enc.detach()
        return x, attns_enc
