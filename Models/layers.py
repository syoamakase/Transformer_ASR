#-*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.modules import MultiHeadAttention, FeedForward

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x2, attn_dec_dec = self.attn_1(x2, x2, x2, trg_mask)
        x = x + self.dropout_1(x2)
        x2 = self.norm_2(x)
        x2, attn_dec_enc = self.attn_2(x2, e_outputs, e_outputs, src_mask)
        x = x + self.dropout_2(x2)
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x, attn_dec_dec, attn_dec_enc

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x2, attn_enc_enc = self.attn(x2, x2, x2, mask)
        x = x + self.dropout_1(x2)
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x, attn_enc_enc
