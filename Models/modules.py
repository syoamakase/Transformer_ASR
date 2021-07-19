#-*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.utils import repeat

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

class CNN_embedding_avepool(nn.Module):
    def __init__(self, hp):
        super().__init__() 
        idim = hp.mel_dim
        out_dim = hp.d_model_e
        cnn_dim = hp.cnn_dim
        self.subsampling_rate = hp.subsampling_rate

        self.cnn_swish = hp.cnn_swish
        if self.cnn_swish:
            self.act = Swish()
        else:
            self.act = nn.ReLU()

        self.l1_flag = hp.l1_flag
        if self.l1_flag:
            self.l1 = nn.Linear(idim, 80) #cnn_dim)
            
        self.conv1 = nn.Conv2d(1, cnn_dim, 3, 1)
        if self.l1_flag:
            hidden_dim = (80-2)//2
        else:
            hidden_dim = (idim-2)//2

        if self.subsampling_rate == 4:
            self.conv2 = nn.Conv2d(cnn_dim, cnn_dim, 3, 1)
            hidden_dim = (hidden_dim-2) // 2 #(idim-2) // 2
        
        hidden_dim *= cnn_dim
        print(f'CNN avepool shape is {hidden_dim}')
        self.out = nn.Linear(hidden_dim, out_dim)

        self.cnn_ln = hp.cnn_ln
        if self.cnn_ln:
            self.ln = nn.LayerNorm(out_dim)

    def forward(self, x, x_mask):
        if self.l1_flag:
            x = self.l1(x).unsqueeze(1)
        else:
            x = x.unsqueeze(1)

        x = self.act(self.conv1(x))
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        x_mask_return = x_mask[:, :, :-3:2]
        if self.subsampling_rate == 4:
            x = self.act(self.conv2(x))
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
            x_mask_return = x_mask_return[:, :, :-3:2]
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if self.cnn_ln:
            x = self.ln(x)

        return x, x_mask_return


class CNN_embedding(nn.Module):
    def __init__(self, hp):
        super().__init__()

        idim = hp.mel_dim
        out_dim = hp.d_model_e
        cnn_dim = hp.cnn_dim
        self.subsampling_rate = hp.subsampling_rate

        self.conv = nn.ModuleList([nn.Conv2d(1, cnn_dim, 3, 2)])
        self.conv.extend([nn.Conv2d(cnn_dim, cnn_dim, 3, 2) for i in range(int(math.log2(self.subsampling_rate)-1))])
        # older version
        #self.conv = torch.nn.Sequential(
        #    nn.Conv2d(1, cnn_dim, 3, 2),
        #    nn.ReLU(),
        #    torch.nn.Conv2d(cnn_dim, cnn_dim, 3, 2),
        #    nn.ReLU(),
        #)
        #self.out = torch.nn.Sequential(
        #    nn.Linear(cnn_dim * (((idim - 1) // 2 - 1) // 2), out_dim),
        #)
        #print(cnn_dim * (((idim - 1) // 2 - 1) // 2))
        hidden_dim = ((idim - 1)// 2)
        #nn.init.kaiming_normal_(self.conv[0].weight.data)
        #if isinstance(self.conv[0].bias, nn.parameter.Parameter):
        #    self.conv[0].bias.data.fill_(0)

        for i in range(int(math.log2(self.subsampling_rate))-1):
            #nn.init.kaiming_normal_(self.conv[i+1].weight.data)
            #if isinstance(self.conv[i+1].bias, nn.parameter.Parameter):
            #    self.conv[i+1].bias.data.fill_(0)
            hidden_dim = (hidden_dim - 1) // 2
        hidden_dim *= cnn_dim
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, x_mask):
        x = x.unsqueeze(1)

        for conv in self.conv:
            x = conv(x)
            x = torch.relu(x)
            x_mask = x_mask[:, :, :-2:2] if x_mask is not None else None

        if x_mask is None:
            return x, None
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c*f))
        return x, x_mask

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
    def __init__(self, heads, d_model, d_model_q=None, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        if d_model_q is not None:
            self.d_model_q = d_model_q
        else:
            self.d_model_q = d_model

        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(self.d_model_q, d_model)
       
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

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

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

class ConvolutionModule(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        causal = False
        kernel_size = 31
        padding = self.calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_model*2, kernel_size=1)
        self.depth_conv1 = DepthwiseConv(in_channels=d_model, out_channels=d_model, kernel_size=kernel_size, padding=padding)
        #self.depth_conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=kernel_size, stride=1, padding=15, groups=d_model)
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.swish = Swish()
        self.pointwise_conv2 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # (B x T x H) -> (B x H x T)

        x = self.layer_norm(x).transpose(1,2)
        x = self.pointwise_conv1(x)
        out, gate = x.chunk(2, dim=1)
        x = out * gate.sigmoid()
        x = self.depth_conv1(x)
        x = self.batch_norm(x)
        x = self.swish(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)

        return x.transpose(1,2)

    def calc_same_padding(self, kernel_size):
        pad = kernel_size // 2
        return (pad, pad - (kernel_size + 1) % 2)

class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()

class DepthwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, groups=in_channels)
        self.conv_out = nn.Conv1d(out_channels, out_channels, 1)

    def forward(self, x):
        x = F.pad(x, self.padding)
        x = self.conv(x)
        return self.conv_out(x)

class FeedForwardConformer(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.swish = Swish()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.swish(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)

        return x

class RelativeMultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)       
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.linear_pos = nn.Linear(d_model, d_model, bias=False)
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, pos_emb, mask=None):

        bs = q.size(0)

        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1,2)
        #q = q.transpose(1,2)
        v = v.transpose(1,2)
        
        # relative pos
        n_batch_pos = pos_emb.shape[0]
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1,2)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1,2)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1,2)

        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2,-1))
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2,-1))
        matrix_bd = self.rel_shift(matrix_bd)

        matrix = matrix_ac+matrix_bd
        scores, attn = self.attention(matrix, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)

        return output, attn

    def rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(2), x.size(3)))
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def attention(self, matrix, v, d_k, mask=None, dropout=None):
    
        scores = matrix / math.sqrt(d_k)
    
        if mask is not None:
            mask = mask.unsqueeze(1)
            attn = scores.masked_fill(mask == 0, -2**15)
            attn = torch.softmax(attn, dim=-1) # (batch, head, time1, time2)
        else:
            attn = F.softmax(attn, dim=-1)
        if dropout is not None:
            attn = dropout(attn)
    
        output = torch.matmul(attn, v)
        return output, attn

class RelativePositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=3000, xscale=1, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.xscale = xscale

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)
        
    def forward(self, x):
        x = x * self.xscale
        seq_len = x.shape[1]
        pe = self.pe[:,:seq_len].to(x.device)

        return self.dropout(x), self.dropout(pe)

class LocationAttention(nn.Module):
    """
    Attention mechanism based on content-based model [Chorowski+, 2015]
    """
    def __init__(self, hp):
        super(LocationAttention, self).__init__()
        self.num_decoder_hidden_nodes = hp.d_model_d
        self.num_encoder_hidden_nodes = hp.d_model_e
        # attention
        self.L_se = nn.Linear(self.num_decoder_hidden_nodes, self.num_decoder_hidden_nodes * 2, bias=False)
        self.L_he = nn.Linear(self.num_encoder_hidden_nodes, self.num_decoder_hidden_nodes * 2)
        self.L_ee = nn.Linear(self.num_decoder_hidden_nodes * 2, 1, bias=False)
        self.L_fe = nn.Linear(10, self.num_decoder_hidden_nodes * 2, bias=False)
        self.F_conv1d = nn.Conv1d(1, 10, 100, stride=1, padding=50, bias=False)

    def forward(self, s, hbatch, alpha, e_mask):
        num_frames = hbatch.size(1)
        tmpconv = self.F_conv1d(alpha)
        tmpconv = tmpconv.transpose(1, 2)[:, :num_frames, :]
        tmpconv = self.L_fe(tmpconv)
        # BxTx2H
        e = torch.tanh(self.L_se(s).unsqueeze(1) + self.L_he(hbatch) + tmpconv)
        # BxT
        e = self.L_ee(e)
        e_nonlin = (e - e.max(1)[0].unsqueeze(1)).exp()
        e_nonlin = e_nonlin * e_mask

        alpha = e_nonlin / e_nonlin.sum(dim=1, keepdim=True)
        g = (alpha * hbatch).sum(dim=1)
        alpha = alpha.transpose(1, 2)

        return g, alpha
