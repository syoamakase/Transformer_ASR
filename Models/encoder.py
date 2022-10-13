#- *- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from Models.utils import repeat
from Models.layers import EncoderLayer, ConformerEncoderLayer
from Models.modules import PositionalEncoder, RelativePositionalEncoder

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
    def __init__(self, hp):
        super().__init__()
        d_model = hp.d_model_e
        self.N = hp.N_e
        self.heads = hp.heads
        self.iter_loss = hp.iter_loss
        batchnorm_momentum = hp.batchnorm_momentum
        dropout = hp.dropout
        xscale = 1 #math.sqrt(d_model)
        self.pe = RelativePositionalEncoder(d_model, xscale=xscale, dropout=dropout)
        self.layers = repeat(self.N, lambda: ConformerEncoderLayer(d_model, self.heads, dropout, batchnorm_momentum=batchnorm_momentum))
        if len(hp.iter_loss) != 0:
            self.iter_out = repeat(len(self.iter_loss), lambda: nn.Linear(d_model, hp.vocab_size))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, mask):
        x, pe = self.pe(src)
        b, t, _ = x.shape
        attns_enc = torch.zeros((b, self.N, self.heads, t, t), device=x.device)
        iter_preds = []
        i_iter_pred = 0
        for i in range(self.N):
            x, attn_enc = self.layers[i](x, pe, mask)
            if i in self.iter_loss:
                pred = self.iter_out[i_iter_pred](x)
                i_iter_pred += 1
                iter_preds.append(pred)
            attns_enc[:,i] = attn_enc.detach()
        return self.norm(x), attns_enc, iter_preds
