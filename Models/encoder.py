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


#class Emformer(nn.Module):
#    def __init__(self, hp,
#        input_dim: int,
#        num_heads: int,
#        ffn_dim: int,
#        num_layers: int,
#        segment_length: int,
#        dropout: float = 0.0,
#        activation: str = "relu",
#        left_context_length: int = 0,
#        right_context_length: int = 0,
#        max_memory_size: int = 0,
#        weight_init_scale_strategy: Optional[str] = "depthwise",
#        tanh_on_mem: bool = False,
#        negative_inf: float = -1e8,
#    ):
#        super().__init__()
#
#        self.dropout = nn.Dropout(dropout)
#        self.memory_op = nn.AvgPool1d(kernel_size=segment_length, stride=segment_length, ceil_mode=True)
#
#        activate_module = self._get_activation_module(activation)
#        self.pos_ff = nn.Sequential(
#            nn.LayerNorm(input_dim),
#            nn.Linear(input_dim, ffn_dim),
#            activation_module,
#            nn.Dropout(dropout),
#            nn.Linear(ffn_dim, input_dim),
#            nn.Dropout(dropout)
#        )
#        self.layer_norm_input = nn.LayerNorm(input_dim)
#        self.layer_norm_output = nn.LayerNorm(input_dim)
#        
#        self.left_contect_length = left_context_length
#        self.segment_length = segment_length
#        self.max_memory_size = max_memory_size
#
#        self.use_mem = max_memory_size > 0
#
#    def forward(nn.Module):
#        pass
#
#
#    def _get_activation_module(activation: str) -> torch.nn.Module:
#        if activation == "relu":
#            return torch.nn.ReLU()
#        elif activation == "gelu":
#            return torch.nn.GELU()
#        elif activation == "silu":
#            return torch.nn.SiLU()
#        else:
#            raise ValueError(f"Unsupported activation {activation}")
#
#    def _init_state(self, batch_size, device):
#        empty_memory = torch.zeros(self.max_memory_size, batch_size, self.input_dim, device=device)
