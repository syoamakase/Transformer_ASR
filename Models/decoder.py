#-*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.utils import repeat
from Models.modules import Embedder, PositionalEncoder
from Models.layers import DecoderLayer

class Decoder(nn.Module):
    def __init__(self, hp):
        super().__init__()
        vocab_size = hp.vocab_size
        d_model = hp.d_model_d
        self.N = hp.N_d
        dropout = hp.dropout
        self.heads = hp.heads

        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = repeat(self.N, lambda: DecoderLayer(d_model, self.heads, dropout))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        b, t1, _ = x.shape
        b, t2, _ = e_outputs.shape
        attns_dec_enc = torch.zeros((b, self.N, self.heads, t1, t2), device=x.device)
        attns_dec_dec = torch.zeros((b, self.N, self.heads, t1, t1), device=x.device)
        for i in range(self.N):
            x, attn_dec_dec, attn_dec_enc = self.layers[i](x, e_outputs, src_mask, trg_mask)
            attns_dec_dec[:,i] = attn_dec_dec.detach()
            attns_dec_enc[:,i] = attn_dec_enc.detach()
        return self.norm(x), attns_dec_dec, attns_dec_enc
