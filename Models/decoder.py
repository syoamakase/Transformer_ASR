# -*- coding: utf-8 -*-
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch import Tensor

import numpy as np
import six
from operator import itemgetter
CTC_SCORING_RATIO = 1.5

from Models.utils import repeat
from Models.modules import Embedder, PositionalEncoder, RelativePositionalEncoder, LocationAttention, MultiHeadAttention, Swish
from Models.layers import DecoderLayer

class Decoder(nn.Module):
    """
    Transformer decoder.
    Args (from hparams.py):
        vocab_size (int): vocabulary size
        d_model_d (int): The dimension of transformer hidden state
        dropout (float): The dropout rate
        heads (int): The number of heads of transformer
        decoder_rel_pos (bool): If True, use Relative positional encoding (future remove)
    """
    def __init__(self, hp):
        super().__init__()
        vocab_size = hp.vocab_size
        d_model = hp.d_model_d
        self.N = hp.N_d
        dropout = hp.dropout
        self.heads = hp.heads
        self.rel_pos = hp.decoder_rel_pos

        self.embed = Embedder(vocab_size, d_model)
        if self.rel_pos:
            self.pe = RelativePositionalEncoder(d_model, xscale=1, dropout=dropout)
        else:
            self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = repeat(self.N, lambda: DecoderLayer(d_model, self.heads, dropout))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        """
        Args:
            trg (torch.Tensor): Target sequence (B x L)
            e_outouts (torch.Tensor): The encoder outoput (B x T x dim of hidden state encoder)
            src_mask (torch.Tensor): The mask of encoder (B x T)
            trg_mask (torch.tensor): The mask of decoder (B x L)

        Returns:
            torch.tensor: Output of decoder (NOT prediction) (B x L x d_model_d)
            torch.tensor: Attention weights of self-attention
            torch.tensor: Attention weights between the decoder and the encoder
        """
        x = self.embed(trg)
        if self.rel_pos:
            x, trg_pe = self.pe(x)
            e_outputs, src_pe = self.pe(e_outputs)            
        else:
            x = self.pe(x)
            src_pe = None
            trg_pe = None
        b, t1, _ = x.shape
        b, t2, _ = e_outputs.shape
        attns_dec_enc = torch.zeros((b, self.N, self.heads, t1, t2), device=x.device)
        attns_dec_dec = torch.zeros((b, self.N, self.heads, t1, t1), device=x.device)
        for i in range(self.N):
            x, attn_dec_dec, attn_dec_enc = self.layers[i](x, e_outputs, src_mask, trg_mask) #, src_pe, trg_pe)
            attns_dec_dec[:,i] = attn_dec_dec.detach()
            attns_dec_enc[:,i] = attn_dec_enc.detach()
        return self.norm(x), attns_dec_dec, attns_dec_enc


class LSTMDecoder(nn.Module):
    """
    LSTM decoder with attention mechanism.
    Args (from hparams.py)
        d_model_d (int): The dimension of decoder hidden state
        d_model_e (int): The dimension of encoder hidden state
        vocab_size (int): Vocabulary size of the output
        multihead (bool): If True, we use multi-head attention for calcuating attention
        weight_dropout (float): Weight dropout rate on LSTM calculation. If None, we use normal Linear.
        norm_lstm (bool):  
    """
    def __init__(self, hp):
        super(LSTMDecoder, self).__init__()
        self.hp = hp
        self.num_decoder_hidden_nodes = hp.d_model_d #hp.num_hidden_nodes_decoder
        self.num_encoder_hidden_nodes = hp.d_model_e
        self.num_classes = hp.vocab_size
        if self.hp.multihead:
            self.att = MultiHeadAttention(4, self.num_encoder_hidden_nodes, self.num_decoder_hidden_nodes, 0.1)
        else:
            self.att = LocationAttention(hp)
        # decoder
        self.L_sy = nn.Linear(self.num_decoder_hidden_nodes, self.num_decoder_hidden_nodes, bias=False)
        self.L_gy = nn.Linear(self.num_encoder_hidden_nodes, self.num_decoder_hidden_nodes)
        self.L_yy = nn.Linear(self.num_decoder_hidden_nodes, self.num_classes)

        #self.L_ys = nn.Embedding(self.num_classes, self.num_decoder_hidden_nodes * 4)
        #self.L_ys = nn.Linear(self.num_classes, self.num_decoder_hidden_nodes * 4 , bias=False)
        #self.L_ss = nn.Linear(self.num_decoder_hidden_nodes, self.num_decoder_hidden_nodes * 4, bias=False)
        #self.L_gs = nn.Linear(self.num_encoder_hidden_nodes, self.num_decoder_hidden_nodes * 4)

        #self.L_ys = nn.Embedding(self.num_classes, self.num_decoder_hidden_nodes)
        #self.L_ys = nn.Linear(self.num_classes, self.num_decoder_hidden_nodes * 4 , bias=False)
        #self.L_ss = nn.Linear(self.num_decoder_hidden_nodes, self.num_decoder_hidden_nodes, bias=False)
        #self.L_gs = nn.Linear(self.num_encoder_hidden_nodes, self.num_decoder_hidden_nodes, bias=False)
        #self.lstm_cell = nn.LSTMCell(self.num_decoder_hidden_nodes, self.num_decoder_hidden_nodes)
        # previous version
        self.L_ys = nn.Embedding(self.num_classes, self.num_decoder_hidden_nodes*4)
        #self.L_ys = nn.Linear(self.num_classes, self.num_decoder_hidden_nodes * 4)
        if self.hp.weight_dropout:
            self.L_ss = WeightDropLinear(self.num_decoder_hidden_nodes, self.num_decoder_hidden_nodes*4, bias=False, weight_dropout=self.hp.weight_dropout)
            self.L_gs = WeightDropLinear(self.num_encoder_hidden_nodes, self.num_decoder_hidden_nodes*4, weight_dropout=self.hp.weight_dropout)
        else:
            self.L_ss = nn.Linear(self.num_decoder_hidden_nodes, self.num_decoder_hidden_nodes*4, bias=False)
            self.L_gs = nn.Linear(self.num_encoder_hidden_nodes, self.num_decoder_hidden_nodes*4)

        #self.norm_y = nn.LayerNorm(self.num_decoder_hidden_nodes)

        if self.hp.norm_lstm:
            self.norm0 = nn.LayerNorm(self.num_decoder_hidden_nodes*4)
            self.norm1 = nn.LayerNorm(self.num_decoder_hidden_nodes*4)
            self.norm2 = nn.LayerNorm(self.num_decoder_hidden_nodes*4)
            #self.norm3 = nn.LayerNorm(self.num_decoder_hidden_nodes)

        if self.hp.swish_lstm:
            self.act = Swish()


    def forward(self, targets, hbatch, src_mask, trg_mask):
        """
        Args:
            targets (torch.LongTensor): Target sequence (B x L)
            hbatch (torch.FloatTensor): Hidden states of the encoder (B x T x hp.d_model_e)
            src_mask (torch.BoolTensor?): Mask for encoder (B x T)
            trg_mask (torch.BoolTensor?): Mask for decoder (B x L) Not used. future remove
        """
        device = hbatch.device
        batch_size = hbatch.size(0)
        num_frames = hbatch.size(1)
        num_labels = targets.size(1)

        e_mask = torch.ones((batch_size, num_frames, 1), requires_grad=False).to(device, non_blocking=True)
        s = torch.zeros((batch_size, self.num_decoder_hidden_nodes), requires_grad=False).to(device, non_blocking=True)
        c = torch.zeros((batch_size, self.num_decoder_hidden_nodes), requires_grad=False).to(device, non_blocking=True)
        youtput = torch.zeros((batch_size, num_labels, self.num_classes), requires_grad=False).to(device, non_blocking=True)
        alpha = torch.zeros((batch_size, 1, num_frames), requires_grad=False).to(device, non_blocking=True)

        e_mask[src_mask.transpose(1,2) is False] = 0.0

        for step in range(num_labels):
            if self.hp.multihead:
                g, alpha = self.att(s.unsqueeze(1), hbatch, hbatch, src_mask)
                g = g.squeeze(1)
            else:
                g, alpha = self.att(s, hbatch, alpha, e_mask)
            # generate
            if self.hp.swish_lstm:
                y = self.L_yy(self.act(self.L_gy(g) + self.L_sy(s)))
            else:
                y = self.L_yy(torch.tanh(self.L_gy(g) + self.L_sy(s)))
            #y = self.L_yy(self.norm_y(self.L_gy(g) + self.L_sy(s)))
            # recurrency calcuate
            if self.hp.norm_lstm:
                rec_input = self.norm0(self.L_ys(targets[:, step])) + self.norm1(self.L_ss(s)) + self.norm2(self.L_gs(g))
            else:
                rec_input = self.L_ys(targets[:, step]) + self.L_ss(s) + self.L_gs(g)

            #c = (torch.sigmoid(forgetgate) * c) + (torch.sigmoid(ingate) * torch.tanh(cellgate))
            #s = torch.sigmoid(outgate) * torch.tanh(s)
            s, c = self._func_lstm(rec_input, c)

            youtput[:, step] = y
        return youtput, None, None
    
    def decode_v2(self, hbatch, src_mask, model_lm=None, lm_weight=0.2, model_lm_2=None, lm_weight_2=0.2, beam_width=10):
        """
        Tedlium 2
        """
        device = hbatch.device
        #import sentencepiece as spm
        #sp = spm.SentencePieceProcessor()
        #sp.Load(self.hp.spm_model)
        batch_size = hbatch.shape[0]
        num_frames = hbatch.shape[1]
        e_mask = torch.ones((batch_size, num_frames, 1), device=device, requires_grad=False)

        #beam_width = 10 #self.hp.beam_width
        max_decoder_seq_len = 200 #self.hp.max_decoder_seq_len
        score_func = 'log_softmax'
        eos_id = 1

        beam_search = {'result': torch.zeros((beam_width, max_decoder_seq_len), device=device, dtype=torch.long),
                       'length': torch.zeros(beam_width).long(),
                       'score': torch.zeros((beam_width), device=device, dtype=torch.float).fill_(0),
                       'c': torch.zeros((beam_width, self.num_decoder_hidden_nodes), device=device),
                       's': torch.zeros((beam_width, self.num_decoder_hidden_nodes), device=device),
                       'alpha': torch.zeros((beam_width, max_decoder_seq_len, num_frames), device=device)}

        beam_results = {'score': torch.zeros((beam_width), device=device, dtype=torch.float).fill_(0),
                        'result': torch.zeros((beam_width, max_decoder_seq_len), device=device, dtype=torch.long),
                        'length': torch.zeros(beam_width).long(),
                        'alpha': torch.zeros((beam_width, max_decoder_seq_len, num_frames), device=device, requires_grad=False)}

        beam_step = 0

        e_mask[src_mask.transpose(1,2) is False] = 0.0
        for seq_step in range(max_decoder_seq_len):
            length_penalty = ((5 + seq_step + 1)**0.9 / (5 + 1)**0.9)
            #length_penalty = ((5 + seq_step + 1)**0.7 / (5 + 1)**0.7)
            cand_seq = copy.deepcopy(beam_search['result'])
            cand_score = copy.deepcopy(beam_search['score'].unsqueeze(1))
            c = copy.deepcopy(beam_search['c'])
            s = copy.deepcopy(beam_search['s'])
            cand_alpha = copy.deepcopy(beam_search['alpha'])
            #TODO: multhead version
            if self.hp.multihead:
                k_v_input = hbatch.expand(beam_width, hbatch.shape[-2], hbatch.shape[-1])
                g, _ = self.att(s, k_v_input, k_v_input, src_mask)
                g = g.squeeze(1)
            else:
                if seq_step == 0:
                    g, alpha = self.att(s, hbatch, cand_alpha[:, seq_step, :].unsqueeze(1), e_mask)
                else:
                    g, alpha = self.att(s, hbatch, cand_alpha[:, seq_step - 1, :].unsqueeze(1), e_mask)
            # generate previous
            #y = self.L_yy(torch.tanh(self.L_gy(g) + self.L_sy(s)))

            # new
            #g, alpha = self.att(s, hbatch, alpha, e_mask)
            # generate
            #y = self.L_yy(torch.tanh(self.L_gy(g) + self.L_sy(s)))
            if self.hp.swish_lstm:
                y = self.L_yy(self.act(self.L_gy(g) + self.L_sy(s)))
            else:
                y = self.L_yy(torch.tanh(self.L_gy(g) + self.L_sy(s)))
            #y = self.L_yy(self.norm_gy(self.L_gy(g)) + self.norm_sy(self.L_sy(s)))

            if score_func == 'log_softmax':
                y = F.log_softmax(y, dim=1)
                if model_lm is not None and seq_step > 0:
                    lm_input = cand_seq[:, :seq_step]
                    lm_score = model_lm(lm_input)[:, -1, :]
                    tmpy = y + lm_weight * F.log_softmax(lm_score, dim=1) #+ 1
                    if model_lm_2 is not None and seq_step > 0:
                        lm_score_2 = model_lm_2(lm_input)[:, -1, :]
                        tmpy += lm_weight_2 * F.log_softmax(lm_score_2, dim=1)
                else:
                    tmpy = y.clone()
            elif score_func == 'softmax':
                y = F.softmax(y, dim=1)
                if model_lm is not None and seq_step:
                    lm_input = cand_seq[:, :seq_step]
                    lm_score = model_lm(lm_input)[:, -1, :]
                    y = y + lm_weight * F.softmax(lm_score, dim=1)
                else:
                    tmpy = y.clone()

            best_scores, best_indices = tmpy.data.topk(beam_width, dim=1)
            scores = cand_score + best_scores + length_penalty #+ 1 #0.5
            tmp_s = torch.zeros((beam_width, self.num_decoder_hidden_nodes), device=device)
            tmp_c = torch.zeros((beam_width, self.num_decoder_hidden_nodes), device=device)

            if seq_step == 0:
                beam_search['score'] = scores[0]
                beam_search['result'][:, 0] = best_indices[0]
                beam_search['length'] += 1
                if not self.hp.multihead:
                    beam_search['alpha'][:, 0, :] = alpha.squeeze(1)
                tmp_s = s
                tmp_c = c
                if self.hp.norm_lstm:
                    rec_input = self.norm0(self.L_ys(best_indices[0])) + self.norm1(self.L_ss(s)) + self.norm2(self.L_gs(g))
                else:
                    rec_input = self.L_ys(best_indices[0]) + self.L_ss(tmp_s) + self.L_gs(g)
                tmps, tmpc = self._func_lstm(rec_input, tmp_c)
                #tmps, tmpc = self.lstm_cell(rec_input, (s, c))
                beam_search['s'] = tmps
                beam_search['c'] = tmpc
            else:
                k_scores, k_ix = scores.reshape(-1).topk(beam_width * 2)
                cand_idx = k_ix // beam_width
                cand_ids = k_ix % beam_width

                num_cand = 0
                i_cand = 0
                tmp_bestidx = torch.zeros((beam_width), dtype=torch.long, device=device)
                tmp_g = torch.zeros((beam_width, self.num_encoder_hidden_nodes), dtype=torch.float, device=device)

                while num_cand < beam_width:
                    if best_indices[cand_idx[i_cand], cand_ids[i_cand]] == eos_id:
                        if cand_seq[cand_idx[i_cand]][0] == 2:
                            beam_results['score'][beam_step] = k_scores[i_cand]
                            beam_results['result'][beam_step] = cand_seq[cand_idx[i_cand]]
                            beam_results['result'][beam_step][seq_step] = best_indices[cand_idx[i_cand], cand_ids[i_cand]]
                            beam_results['length'][beam_step] = seq_step + 1
                            if not self.hp.multihead:
                                beam_results['alpha'][beam_step] = cand_alpha[cand_idx[i_cand], :, :]
                                beam_results['alpha'][beam_step][seq_step] = alpha[cand_idx[i_cand]].squeeze(0)
                            beam_step += 1
                        i_cand += 1
                    else:
                        beam_search['score'][num_cand] = k_scores[i_cand]
                        beam_search['result'][num_cand] = cand_seq[cand_idx[i_cand]]
                        beam_search['result'][num_cand][seq_step] = best_indices[cand_idx[i_cand], cand_ids[i_cand]]
                        beam_search['length'][num_cand] += 1
                        tmp_bestidx[num_cand] = best_indices[cand_idx[i_cand], cand_ids[i_cand]]
                        if not self.hp.multihead:
                            beam_search['alpha'][num_cand] = cand_alpha[cand_idx[i_cand], :, :]
                            beam_search['alpha'][num_cand][seq_step] = alpha[cand_idx[i_cand]].squeeze(0)
                        tmp_s[num_cand] = s[cand_idx[i_cand]]
                        tmp_c[num_cand] = c[cand_idx[i_cand]]
                        tmp_g[num_cand] = g[cand_idx[i_cand]]

                        i_cand += 1
                        num_cand += 1

                    if beam_step >= beam_width:
                        break

                if self.hp.norm_lstm:
                    rec_input = self.norm0(self.L_ys(tmp_bestidx)) + self.norm1(self.L_ss(tmp_s)) + self.norm2(self.L_gs(tmp_g))
                else:
                    rec_input = self.L_ys(tmp_bestidx) + self.L_ss(tmp_s) + self.L_gs(tmp_g)
                tmps, tmpc = self._func_lstm(rec_input, tmp_c)
                # recurrency calcuate
                beam_search['s'] = tmps
                beam_search['c'] = tmpc

                if beam_step >= beam_width:
                    break
        best_idx = beam_results['score'].argmax()
        length = beam_results['length'][best_idx]
        results = beam_results['result'][best_idx][:length].cpu().tolist()
        attention = beam_results['alpha'][best_idx, :length]

        import matplotlib.pyplot as plt
        import sentencepiece as spm
        attention = attention.cpu().numpy()
        sp = spm.SentencePieceProcessor()
        sp.Load(self.hp.spm_model)

        return results
    
    def decode_v3(self, hbatch, src_mask, model_lm=None, lm_weight=0.2, model_lm_2=None, lm_weight_2=0.2, beam_width=10, version=3):
        """
        LibriSpeech
        """
        device = hbatch.device
        batch_size = hbatch.shape[0]
        num_frames = hbatch.shape[1]
        e_mask = torch.ones((batch_size, num_frames, 1), device=device, requires_grad=False)

        max_decoder_seq_len = 200 #self.hp.max_decoder_seq_len
        score_func = 'log_softmax'
        eos_id = 1

        beam_search = {'result': torch.zeros((beam_width, max_decoder_seq_len), device=device, dtype=torch.long),
                       'length': torch.zeros(beam_width).long(),
                       'score': torch.zeros((beam_width), device=device, dtype=torch.float).fill_(0),
                       'c': torch.zeros((beam_width, self.num_decoder_hidden_nodes), device=device),
                       's': torch.zeros((beam_width, self.num_decoder_hidden_nodes), device=device),
                       'alpha': torch.zeros((beam_width, max_decoder_seq_len, num_frames), device=device)}

        beam_results = {'score': torch.zeros((beam_width), device=device, dtype=torch.float).fill_(0),
                        'result': torch.zeros((beam_width, max_decoder_seq_len), device=device, dtype=torch.long),
                        'length': torch.zeros(beam_width).long(),
                        'alpha': torch.zeros((beam_width, max_decoder_seq_len, num_frames), device=device, requires_grad=False)}

        beam_step = 0

        e_mask[src_mask.transpose(1,2) is False] = 0.0
        for seq_step in range(max_decoder_seq_len):
            length_penalty = 0 #((5 + seq_step + 1)**0.9 / (5 + 1)**0.9)
            
            cand_seq = copy.deepcopy(beam_search['result'])
            cand_score = copy.deepcopy(beam_search['score'].unsqueeze(1))
            c = copy.deepcopy(beam_search['c'])
            s = copy.deepcopy(beam_search['s'])
            cand_alpha = copy.deepcopy(beam_search['alpha'])
            #TODO: multhead version
            if self.hp.multihead:
                k_v_input = hbatch.expand(beam_width, hbatch.shape[-2], hbatch.shape[-1])
                g, _ = self.att(s, k_v_input, k_v_input, src_mask)
                g = g.squeeze(1)
            else:
                if seq_step == 0:
                    g, alpha = self.att(s, hbatch, cand_alpha[:, seq_step, :].unsqueeze(1), e_mask)
                else:
                    g, alpha = self.att(s, hbatch, cand_alpha[:, seq_step - 1, :].unsqueeze(1), e_mask)
            # generate previous
            #y = self.L_yy(torch.tanh(self.L_gy(g) + self.L_sy(s)))

            # new
            #g, alpha = self.att(s, hbatch, alpha, e_mask)
            # generate
            #y = self.L_yy(torch.tanh(self.L_gy(g) + self.L_sy(s)))
            if self.hp.swish_lstm:
                y = self.L_yy(self.act(self.L_gy(g) + self.L_sy(s)))
            else:
                y = self.L_yy(torch.tanh(self.L_gy(g) + self.L_sy(s)))
            #y = self.L_yy(self.norm_gy(self.L_gy(g)) + self.norm_sy(self.L_sy(s)))

            if score_func == 'log_softmax':
                y = F.log_softmax(y, dim=1) + 3
                if model_lm is not None and seq_step > 0:
                    lm_input = cand_seq[:, :seq_step]
                    lm_score = model_lm(lm_input)[:, -1, :]
                    tmpy = y + lm_weight * F.log_softmax(lm_score, dim=1)
                    if model_lm_2 is not None and seq_step > 0:
                        lm_score_2 = model_lm_2(lm_input)[:, -1, :]
                        tmpy += lm_weight_2 * F.log_softmax(lm_score_2, dim=1)
                else:
                    tmpy = y.clone()
            elif score_func == 'softmax':
                y = F.softmax(y, dim=1)
                if model_lm is not None and seq_step:
                    lm_input = cand_seq[:, :seq_step]
                    lm_score = model_lm(lm_input)[:, -1, :]
                    y = y + lm_weight * F.softmax(lm_score, dim=1)
                else:
                    tmpy = y.clone()

            best_scores, best_indices = tmpy.data.topk(beam_width, dim=1)
            scores = cand_score + best_scores + length_penalty
            tmp_s = torch.zeros((beam_width, self.num_decoder_hidden_nodes), device=device)
            tmp_c = torch.zeros((beam_width, self.num_decoder_hidden_nodes), device=device)

            if seq_step == 0:
                beam_search['score'] = scores[0]
                beam_search['result'][:, 0] = best_indices[0]
                beam_search['length'] += 1
                if not self.hp.multihead:
                    beam_search['alpha'][:, 0, :] = alpha.squeeze(1)
                tmp_s = s
                tmp_c = c
                if self.hp.norm_lstm:
                    rec_input = self.norm0(self.L_ys(best_indices[0])) + self.norm1(self.L_ss(s)) + self.norm2(self.L_gs(g))
                else:
                    rec_input = self.L_ys(best_indices[0]) + self.L_ss(tmp_s) + self.L_gs(g)
                tmps, tmpc = self._func_lstm(rec_input, tmp_c)
                #tmps, tmpc = self.lstm_cell(rec_input, (s, c))
                beam_search['s'] = tmps
                beam_search['c'] = tmpc
            else:
                k_scores, k_ix = scores.reshape(-1).topk(beam_width * 2)
                cand_idx = k_ix // beam_width
                cand_ids = k_ix % beam_width

                num_cand = 0
                i_cand = 0
                tmp_bestidx = torch.zeros((beam_width), dtype=torch.long, device=device)
                tmp_g = torch.zeros((beam_width, self.num_encoder_hidden_nodes), dtype=torch.float, device=device)

                while num_cand < beam_width:
                    if best_indices[cand_idx[i_cand], cand_ids[i_cand]] == eos_id:
                        if cand_seq[cand_idx[i_cand]][0] == 2:
                            beam_results['score'][beam_step] = k_scores[i_cand]
                            beam_results['result'][beam_step] = cand_seq[cand_idx[i_cand]]
                            beam_results['result'][beam_step][seq_step] = best_indices[cand_idx[i_cand], cand_ids[i_cand]]
                            beam_results['length'][beam_step] = seq_step + 1
                            if not self.hp.multihead:
                                beam_results['alpha'][beam_step] = cand_alpha[cand_idx[i_cand], :, :]
                                beam_results['alpha'][beam_step][seq_step] = alpha[cand_idx[i_cand]].squeeze(0)
                            beam_step += 1
                        i_cand += 1
                    else:
                        beam_search['score'][num_cand] = k_scores[i_cand]
                        beam_search['result'][num_cand] = cand_seq[cand_idx[i_cand]]
                        beam_search['result'][num_cand][seq_step] = best_indices[cand_idx[i_cand], cand_ids[i_cand]]
                        beam_search['length'][num_cand] += 1
                        tmp_bestidx[num_cand] = best_indices[cand_idx[i_cand], cand_ids[i_cand]]
                        if not self.hp.multihead:
                            beam_search['alpha'][num_cand] = cand_alpha[cand_idx[i_cand], :, :]
                            beam_search['alpha'][num_cand][seq_step] = alpha[cand_idx[i_cand]].squeeze(0)
                        tmp_s[num_cand] = s[cand_idx[i_cand]]
                        tmp_c[num_cand] = c[cand_idx[i_cand]]
                        tmp_g[num_cand] = g[cand_idx[i_cand]]

                        i_cand += 1
                        num_cand += 1

                    if beam_step >= beam_width:
                        break

                if self.hp.norm_lstm:
                    rec_input = self.norm0(self.L_ys(tmp_bestidx)) + self.norm1(self.L_ss(tmp_s)) + self.norm2(self.L_gs(tmp_g))
                else:
                    rec_input = self.L_ys(tmp_bestidx) + self.L_ss(tmp_s) + self.L_gs(tmp_g)
                tmps, tmpc = self._func_lstm(rec_input, tmp_c)
                # recurrency calcuate
                beam_search['s'] = tmps
                beam_search['c'] = tmpc

                if beam_step >= beam_width:
                    break
        best_idx = beam_results['score'].argmax()
        length = beam_results['length'][best_idx]
        results = beam_results['result'][best_idx][:length].cpu().tolist()
        attention = beam_results['alpha'][best_idx, :length]

        import matplotlib.pyplot as plt
        import sentencepiece as spm
        attention = attention.cpu().numpy()
        sp = spm.SentencePieceProcessor()
        sp.Load(self.hp.spm_model)

        return results


    @staticmethod
    def _func_lstm(x, c):
        ingate, forgetgate, cellgate, outgate = x.chunk(4, 1)
        c_next = (torch.sigmoid(forgetgate) * c) + (torch.sigmoid(ingate) * torch.tanh(cellgate))
        h = torch.sigmoid(outgate) * torch.tanh(c_next)
        return h, c_next

    def _plot_attention(self, attention, label=None):
        import matplotlib.pyplot as plt
        import sentencepiece as spm
        attention = attention.cpu().numpy()
        sp = spm.SentencePieceProcessor()
        sp.Load(self.hp.spm_model)
        return 
        

    def joint_ctc_decoding(self, hbatch, src_mask, ctc_out, model_lm, lm_weight, ctc_weight=0.4, beam_width=10):
        #  hbatch, src_mask, model_lm=None, lm_weight=0.2, model_lm_2=None, lm_weight_2=0.2, beam_width=10):
    # def recog(self, enc_output, recog_args, char_list=None, rnnlm=None, use_jit=False):
        """Recognize input speech of each speaker.
        :param ndnarray enc_output: encoder outputs (B, T, D) or (T, D)
        :param Namespace recog_args: argment Namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        device = hbatch.device
        if ctc_weight >= 0.0:
            lpz = F.log_softmax(ctc_out, dim=-1)
            lpz = lpz.squeeze(0)
        else:
            lpz = [None]

        h = hbatch.squeeze(0)
        eos = 1
        eos_id = 1
        num_encs = 1
        length_penalty = 0 #((5 + seq_step + 1)**0.9 / (5 + 1)**0.9)
        score_func = 'log_softmax'
        max_decoder_seq_len = 200
        #
        batch_size = hbatch.shape[0]
        num_frames = hbatch.shape[1]
        e_mask = torch.ones((batch_size, num_frames, 1), device=device, requires_grad=False)

        ## logging.info("input lengths: " + str(h.size(0)))
        # search parms
        beam = beam_width
        penalty = 0.0 ## recog_args.penalty
        ## ctc_weight = recog_args.ctc_weight
        beam = {'result': [],
                'length': torch.zeros(1).long(),
                'score': torch.zeros((1), device=device, dtype=torch.float).fill_(0),
                'c': torch.zeros((1, self.num_decoder_hidden_nodes), device=device),
                's': torch.zeros((1, self.num_decoder_hidden_nodes), device=device),
                'alpha': torch.zeros((1, num_frames), device=device)}

        if lpz[0] is not None:
            # x, blank, eos, xp
            ctc_prefix_score = CTCPrefixScore(lpz.cpu().detach().numpy(), blank=0, eos=1, xp=np)
            # import pdb; pdb.set_trace()
            beam["ctc_state_prev"] = ctc_prefix_score.initial_state()
            beam["ctc_score_prev"] = 0.0
            if ctc_weight != 1.0:
                # pre-pruning based on attention scores
                ctc_beam = min(lpz[0].shape[-1], int(beam_width * CTC_SCORING_RATIO))
            else:
                ctc_beam = lpz[0].shape[-1]
        else:
            ctc_beam = beam_width

        beams = [beam]
        decoded = False
        results = []

        # import pdb; pdb.set_trace()
        e_mask[src_mask.transpose(1,2) is False] = 0.0
        for seq_step in range(max_decoder_seq_len):
            new_beams = []

            for beam in beams:
                cand_seq = torch.tensor(beam['result'], dtype=torch.float32, device=device)
                cand_score = beam['score'].unsqueeze(1)
                c = beam['c']
                s = beam['s']
                cand_alpha = beam['alpha']
                #TODO: multhead version
                if self.hp.multihead:
                    k_v_input = hbatch.expand(beam_width, hbatch.shape[-2], hbatch.shape[-1])
                    g, _ = self.att(s, k_v_input, k_v_input, src_mask)
                    g = g.squeeze(1)
                else:
                    if seq_step == 0:
                        g, alpha = self.att(s, hbatch, cand_alpha.unsqueeze(1), e_mask)
                    else:
                        # import pdb;pdb.set_trace()
                        g, alpha = self.att(s, hbatch, cand_alpha.unsqueeze(1), e_mask)

                if self.hp.swish_lstm:
                    y = self.L_yy(self.act(self.L_gy(g) + self.L_sy(s)))
                else:
                    y = self.L_yy(torch.tanh(self.L_gy(g) + self.L_sy(s)))

                if score_func == 'log_softmax':
                    scores_att = F.log_softmax(y, dim=1)

                # scores = scores_att

                if seq_step > 0:
                    # import pdb; pdb.set_trace()
                    lm_input = cand_seq.unsqueeze(0).long()
                    lm_score = model_lm(lm_input)[:, -1, :]
                    scores_att += lm_weight * F.log_softmax(lm_score, dim=1)
                    # scores += lm_weight * F.log_softmax(lm_score, dim=1)
                
                if lpz[0] is not None and seq_step > 0:
                    best_scores, best_indices = torch.topk(scores_att, ctc_beam, dim=1)
                
                    # import pdb; pdb.set_trace()
                    ctc_scores, ctc_states = ctc_prefix_score(beam["result"],
                        best_indices[0].cpu().numpy(), beam['ctc_state_prev'])

                    scores = (1 - ctc_weight) * scores_att[:, best_indices[0]] + ctc_weight * torch.from_numpy(ctc_scores - beam["ctc_score_prev"]).to(device)

                    best_scores, ids_best_sel = torch.topk(
                        scores, beam_width, dim=1
                    )
                    best_indices = best_indices[:, ids_best_sel[0]]
                else:
                    # TODO: 
                    best_scores, best_indices = torch.topk(scores_att, beam_width, dim=1)
                    # local_scores = y[:, best_indices[0]]


                for beam_i in range(beam_width):
                    new_beam = {}

                    tmp_s = torch.zeros((1, self.num_decoder_hidden_nodes), device=device)
                    tmp_c = torch.zeros((1, self.num_decoder_hidden_nodes), device=device)

                    new_beam['score'] = beam['score'] + float(best_scores[0, beam_i])
                    # import pdb;pdb.set_trace()
                    new_beam['result'] = beam["result"] + [int(best_indices[0, beam_i])]
                    new_beam['length'] = beam['length'] + 1
                    if not self.hp.multihead:
                        new_beam['alpha'] = alpha.squeeze(1)
                    tmp_s = s
                    tmp_c = c
                    if self.hp.norm_lstm:
                        rec_input = self.norm0(self.L_ys(best_indices[0])) + self.norm1(self.L_ss(s)) + self.norm2(self.L_gs(g))
                    else:
                        rec_input = self.L_ys(best_indices[0]) + self.L_ss(tmp_s) + self.L_gs(g)
                    tmps, tmpc = self._func_lstm(rec_input, tmp_c)
                    #tmps, tmpc = self.lstm_cell(rec_input, (s, c))
                    new_beam['s'] = tmps
                    new_beam['c'] = tmpc
                    if ctc_weight >= 0.0:
                        if seq_step != 0:
                            new_beam["ctc_state_prev"] = ctc_states[ids_best_sel[0, beam_i]]
                            new_beam["ctc_score_prev"] = ctc_scores[ids_best_sel[0, beam_i]]
                        else:
                            new_beam["ctc_state_prev"] = ctc_prefix_score.initial_state()
                            new_beam["ctc_score_prev"] = 0.0                           

                    new_beams.append(new_beam)

            beams = sorted(new_beams, key=lambda x: x["score"], reverse=True)[:beam_width]

            remained_beams = []
            for beam in beams:
                if beam["result"][-1] == eos_id:
                    if len(beam["result"]) <= 1:
                        continue

                    len_p = 0.0
                    score = beam["score"] + len_p * len(beam["result"][1:-1])
                    results.append((beam["result"], score))
                    if len(results) >= beam_width:
                        decoded = True
                        break
                else:
                    remained_beams.append(beam)

                if decoded:
                    break
                
            beams = remained_beams

        results = sorted(results, key=itemgetter(1), reverse=True)
        import pdb;pdb.set_trace()
        results_return = results[0][0]

        return results_return       


    def recognize(self, x, recog_args, char_list=None, rnnlm=None, use_jit=False):
        """Recognize input speech of each speaker.
        :param ndnarray x: input acoustic feature (B, T, D) or (T, D)
        :param Namespace recog_args: argment Namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        # Encoder
        enc_output = self.encode(x)

        # Decoder
        nbest_hyps = []
        for enc_out in enc_output:
            nbest_hyps.append(
                self.recog(enc_out, recog_args, char_list, rnnlm, use_jit)
            )
        return nbest_hyps

class TransducerDecoder(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        self.num_decoder_hidden_nodes = hp.d_model_d #* 2 #hp.num_hidden_nodes_decoder
        self.num_encoder_hidden_nodes = hp.d_model_e
        self.num_classes = hp.vocab_size
        self.blank = 0
        self.swish = hp.decoder_swish
        self.use_lm_loss = hp.use_lm_loss
        self.num_decoder = hp.n_model_d
        self.use_aux_transducer_loss = hp.use_aux_transducer_loss
        self.use_symm_kl_div_loss = hp.use_symm_kl_div_loss

        # previous
        #self.embed = nn.Embedding(self.num_classes, self.num_classes-1, padding_idx=self.blank)
        #self.embed.weight.data[1:] = torch.eye(self.num_classes-1)
        #self.embed.weight.requires_grad = False
        #self.decoder = nn.LSTM(self.num_classes-1, self.num_decoder_hidden_nodes, 2, batch_first=True, dropout=0.1)
        #self.decoder = nn.LSTM(self.num_classes-1, self.num_decoder_hidden_nodes, 1, batch_first=True, dropout=0.1)

        # new
        self.embed = nn.Embedding(self.num_classes, self.num_decoder_hidden_nodes, padding_idx=self.blank)
        #self.decoder = nn.LSTM(self.num_decoder_hidden_nodes, self.num_decoder_hidden_nodes, 1, batch_first=True, dropout=0.1)
        self.decoder = nn.LSTM(self.num_decoder_hidden_nodes, self.num_decoder_hidden_nodes, self.num_decoder, batch_first=True, dropout=0.1)

        #self.fc1 = nn.Linear(self.num_encoder_hidden_nodes + self.num_decoder_hidden_nodes, self.num_decoder_hidden_nodes)
        self.fc1_enc = WeightDropLinear(self.num_encoder_hidden_nodes, self.num_decoder_hidden_nodes, bias=False, weight_dropout=0.2)
        self.fc1_dec = WeightDropLinear(self.num_decoder_hidden_nodes, self.num_decoder_hidden_nodes, bias=False, weight_dropout=0.2)
        #self.fc1 = nn.Linear(self.num_encoder_hidden_nodes + self.num_decoder_hidden_nodes, self.num_decoder_hidden_nodes)
        self.fc2 = nn.Linear(self.num_decoder_hidden_nodes, self.num_classes)

        if self.use_lm_loss:
            self.fc_lm = nn.Linear(self.num_decoder_hidden_nodes, self.num_classes)

        if self.use_aux_transducer_loss:
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(self.num_encoder_hidden_nodes, self.num_decoder_hidden_nodes),
                torch.nn.LayerNorm(self.num_decoder_hidden_nodes),
                torch.nn.Dropout(p=0.1),
                torch.nn.ReLU(),
                torch.nn.Linear(self.num_decoder_hidden_nodes, self.num_decoder_hidden_nodes),
            )
        
    def joint(self, f, g):
        ## concat 
        #out = torch.cat((f, g), dim=-1)
        #out = torch.relu(self.fc1(out))
        ## add
        if self.swish:
            out = F.silu(self.fc1_enc(f) + self.fc1_dec(g))
        else:
            out = torch.relu(self.fc1_enc(f) + self.fc1_dec(g))
        return self.fc2(out)

    def forward(self, targets, hbatch, hbatch_aux=None):
        assert (self.use_aux_transducer_loss and hbatch_aux is not None) or (not self.use_aux_transducer_loss and hbatch_aux is None)
        device = hbatch.device
        zero = torch.zeros((targets.shape[0], 1), device=device, requires_grad=True).long()
        ymat = torch.cat((zero, targets), dim=1)

        ymat = self.embed(ymat)
        dec_out, _ = self.decoder(ymat)

        hbatch = hbatch.unsqueeze(dim=2)
        ymat = dec_out.unsqueeze(dim=1)
        # expand 
        sz = [max(i, j) for i, j in zip(hbatch.size()[:-1], ymat.size()[:-1])]
        hbatch = hbatch.expand(torch.Size(sz+[hbatch.shape[-1]]));
        ymat = ymat.expand(torch.Size(sz+[ymat.shape[-1]]))
        out = self.joint(hbatch, ymat)
        # torch.Size([6, 391, 65, 10000])
        if self.use_lm_loss:
            lm_outputs = self.fc_lm(dec_out)
            out = (out, lm_outputs)

        if self.use_aux_transducer_loss:
            for p in self.fc1_enc:
                p.requires_grad = False
            for i, aux_enc_out in enumerate(hbatch_aux):
                pass
            
        return out

    def decode(self, hbatch):
        device = hbatch.device
        vy = torch.tensor([[0]], dtype=torch.long, device=device).view(1, 1)
        y, h = self.decoder(self.embed(vy))
        y_seq = []
        logp = 0
        for enc in hbatch[0]:
            ytu = self.joint(enc, y[0][0])
            out = F.log_softmax(ytu, dim=0)
            p, pred = torch.max(out, dim=0)
            pred = int(pred)
            logp += float(p)
            if pred != self.blank:
                y_seq.append(pred)
                vy.data[0][0] = pred
                y, h = self.decoder(self.embed(vy), h)

        return y_seq# , -logp

    def decode_v2(self, hbatch, src_mask, model_lm=None, lm_weight=0.2, model_lm_2=None, lm_weight_2=0.2, beam_width=10):
        device = hbatch.device
        #vy = torch.tensor([[0]], dtype=torch.long, device=device).view(1, 1)
        #beam_search = {'result': torch.zeros((beam_width, max_decoder_seq_len), device=device, dtype=torch.long),
        #               'length': torch.zeros(beam_width).long(),
        #               'score': torch.zeros((beam_width), device=device, dtype=torch.float).fill_(0),
        #               'h': torch.zeros((beam_width, self.num_decoder_hidden_nodes), device=device)}

        k_range = min(beam_width, self.num_classes)
        nbest = 1

        beam_results = [{'score': 0.0,
                        'result': [self.blank],
                        'cache': None}]

        B_hyps = [{'score': 0.0, 'result': [self.blank], 'cache': None}]
        for enc in hbatch[0]:
            A_hyps = B_hyps
            B_hyps = []

            while True:
                new_hyp = max(A_hyps, key=lambda x: x['score'])
                A_hyps.remove(new_hyp)

                ys = torch.tensor(new_hyp['result']).long().unsqueeze(0).to(device)
                ret = torch.ones(len(new_hyp['result']), len(new_hyp['result']), device=device, dtype=torch.bool)
                ys_mask = torch.tril(ret, out=ret).unsqueeze(0)

                y, h = self.decoder(self.embed(ys), new_hyp['cache'])

                #import pdb; pdb.set_trace()
                ytu = torch.log_softmax(self.joint(enc, y[0][0]),  dim=0) ## TODO: lm score

                ytu = ytu.topk(k_range, dim=-1)

                #import pdb;pdb.set_trace()
                for score, k in zip(*ytu):
                    beam_hyp = {'score': new_hyp['score'] + float(score),
                                'result': new_hyp['result'][:],
                                'cache': new_hyp['cache']}

                    if k == self.blank:
                        B_hyps.append(beam_hyp)
                    else:
                        beam_hyp['result'].append(int(k))
                        beam_hyp['cache'] = h
                        A_hyps.append(beam_hyp)

                if len(B_hyps) >= k_range:
                    break

        nbest_hyps = sorted(B_hyps, key=lambda x: x['score'] / len(x['result']), reverse=True)[:nbest]
        import pdb; pdb.set_trace()
                        
        return nbest_hyps    


class WeightDropLinear(torch.nn.Linear):
    """
    Wrapper around :class:`torch.nn.Linear` that adds ``weight_dropout`` named argument.

    Args:
        weight_dropout (float): The probability a weight will be dropped.
    """

    def __init__(self, *args, weight_dropout=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        weights = ['weight']
        _weight_drop(self, weights, weight_dropout)


def _weight_drop(module, weights, dropout):
    """
    Helper for `WeightDrop`.
    """

    for name_w in weights:
        w = getattr(module, name_w)
        del module._parameters[name_w]
        module.register_parameter(name_w + '_raw', Parameter(w))

    original_module_forward = module.forward

    def forward(*args, **kwargs):
        for name_w in weights:
            raw_w = getattr(module, name_w + '_raw')
            w = torch.nn.functional.dropout(raw_w, p=dropout, training=module.training)
            setattr(module, name_w, w)

        return original_module_forward(*args, **kwargs)

    setattr(module, 'forward', forward)


### copy from  https://github.com/espnet/espnet/blob/0473be37bd96d0640057447d27d3789c31490221/espnet/nets/ctc_prefix_score.py#L273
class CTCPrefixScore(object):
    """Compute CTC label sequence scores
    which is based on Algorithm 2 in WATANABE et al.
    "HYBRID CTC/ATTENTION ARCHITECTURE FOR END-TO-END SPEECH RECOGNITION,"
    but extended to efficiently compute the probablities of multiple labels
    simultaneously
    """

    def __init__(self, x, blank, eos, xp):
        self.xp = xp
        self.logzero = -10000000000.0
        self.blank = blank
        self.eos = eos
        self.input_length = len(x)
        self.x = x

    def initial_state(self):
        """Obtain an initial CTC state
        :return: CTC state
        """
        # initial CTC state is made of a frame x 2 tensor that corresponds to
        # r_t^n(<sos>) and r_t^b(<sos>), where 0 and 1 of axis=1 represent
        # superscripts n and b (non-blank and blank), respectively.
        r = self.xp.full((self.input_length, 2), self.logzero, dtype=np.float32)
        r[0, 1] = self.x[0, self.blank]

        for i in six.moves.range(1, self.input_length):
            r[i, 1] = r[i - 1, 1] + self.x[i, self.blank]
        return r

    def __call__(self, y, cs, r_prev):
        """Compute CTC prefix scores for next labels
        :param y     : prefix label sequence
        :param cs    : array of next labels
        :param r_prev: previous CTC state
        :return ctc_scores, ctc_states
        """
        # initialize CTC states
        output_length = len(y) - 1  # ignore sos
        # new CTC states are prepared as a frame x (n or b) x n_labels tensor
        # that corresponds to r_t^n(h) and r_t^b(h).
        r = self.xp.ndarray((self.input_length, 2, len(cs)), dtype=np.float32)
        xs = self.x[:, cs]
        if output_length == 0:
            r[0, 0] = xs[0]
            r[0, 1] = self.logzero
        else:
            r[output_length - 1] = self.logzero

        # prepare forward probabilities for the last label
        r_sum = self.xp.logaddexp(
            r_prev[:, 0], r_prev[:, 1]
        )  # log(r_t^n(g) + r_t^b(g))
        last = y[-1]
        if output_length > 0 and last in cs:
            log_phi = self.xp.ndarray((self.input_length, len(cs)), dtype=np.float32)
            for i in six.moves.range(len(cs)):
                log_phi[:, i] = r_sum if cs[i] != last else r_prev[:, 1]
        else:
            log_phi = r_sum

        # compute forward probabilities log(r_t^n(h)), log(r_t^b(h)),
        # and log prefix probabilities log(psi)
        start = max(output_length, 1)
        log_psi = r[start - 1, 0]
        for t in six.moves.range(start, self.input_length):
            r[t, 0] = self.xp.logaddexp(r[t - 1, 0], log_phi[t - 1]) + xs[t]
            r[t, 1] = (
                self.xp.logaddexp(r[t - 1, 0], r[t - 1, 1]) + self.x[t, self.blank]
            )
            log_psi = self.xp.logaddexp(log_psi, log_phi[t - 1] + xs[t])

        # get P(...eos|X) that ends with the prefix itself
        eos_pos = self.xp.where(cs == self.eos)[0]
        if len(eos_pos) > 0:
            log_psi[eos_pos] = r_sum[-1]  # log(r_T^n(g) + r_T^b(g))

        # exclude blank probs
        blank_pos = self.xp.where(cs == self.blank)[0]
        if len(blank_pos) > 0:
            log_psi[blank_pos] = self.logzero

        # return the log prefix probability and CTC states, where the label axis
        # of the CTC states is moved to the first axis to slice it easily
        return log_psi, self.xp.rollaxis(r, 2)
