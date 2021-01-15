#-*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=3000, dropout=0.1, pe_alpha=False, xscale=None):
        super().__init__()
        self.d_model = d_model
        self.pe_alpha = pe_alpha
        self.dropout = nn.Dropout(dropout)
        self.xscale = xscale if xscale else math.sqrt(d_model)

        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        x = x * self.xscale
        seq_len = x.size(1)
        pe = self.pe[:,:seq_len].to(x.device)
        if self.pe_alpha:
            x = x + self.alpha * pe
        else:
            x = x + pe
        return self.dropout(x)

class CNN_embedding(nn.Module):
    def __init__(self, hp):
        super().__init__()
        
        idim = hp.mel_dim
        out_dim = hp.d_model_e
        cnn_dim = hp.cnn_dim

        self.conv = torch.nn.Sequential(
            nn.Conv2d(1, cnn_dim, 3,2),
            nn.ReLU(),
            torch.nn.Conv2d(cnn_dim, cnn_dim, 3, 2),
            nn.ReLU()
        )
        self.out = torch.nn.Sequential(
            nn.Linear(cnn_dim * (((idim - 1) // 2 - 1) // 2), out_dim),
        )

    def forward(self, x, x_mask):
        x = x.unsqueeze(1)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c*f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:2]

class Embedder(nn.Module):
    # copy
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embed(x)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()

        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

        nn.init.xavier_uniform_(self.linear_1.weight,
            gain=nn.init.calculate_gain('linear'))
        nn.init.xavier_uniform_(self.linear_2.weight,
            gain=nn.init.calculate_gain('linear'))

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
       
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):

        bs = q.size(0)

        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        # calculate attention using function we will define next
        scores, attn = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)

        return output, attn

def attention(q, k, v, d_k, mask=None, dropout=None):

    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e4)
        scores = torch.softmax(scores, dim=-1) # (batch, head, time1, time2)
    else:
        scores = F.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output, scores
