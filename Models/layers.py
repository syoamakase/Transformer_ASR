#-*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.modules import MultiHeadAttention, FeedForward, FeedForwardConformer, ConvolutionModule, RelativeMultiHeadAttention

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, d_model, d_model_q=d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, d_model_q=d_model, dropout=dropout)
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
        self.attn = MultiHeadAttention(heads, d_model, d_model_q=d_model, dropout=dropout)
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

class ConformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.ff_1 = FeedForwardConformer(d_model, d_ff=d_model*4, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.attn = RelativeMultiHeadAttention(heads, d_model, dropout=dropout)
        self.conv_module = ConvolutionModule(d_model, dropout=dropout)
        self.ff_2 = FeedForwardConformer(d_model, d_ff=d_model*4, dropout=dropout)

        self.dropout_1 = nn.Dropout(dropout)

    def forward(self, x, pe, mask):
        x = x + 0.5 * self.ff_1(x)
        res = x
        x = self.norm(x)
        x, attn_enc_enc = self.attn(x,x,x,pe,mask)
        x = res + self.dropout_1(x)
        x = x + self.conv_module(x)
        x = x + 0.5 * self.ff_2(x)
        return x, attn_enc_enc
