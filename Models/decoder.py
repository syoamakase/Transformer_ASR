#-*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.utils import repeat
from Models.modules import Embedder, PositionalEncoder, RelativePositionalEncoder, Attention
from Models.layers import DecoderLayer

class Decoder(nn.Module):
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
        self.layers = repeat(self.N, lambda: DecoderLayer(d_model, self.heads, self.rel_pos, dropout))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, trg, e_outputs, src_mask, trg_mask):
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
            x, attn_dec_dec, attn_dec_enc = self.layers[i](x, e_outputs, src_mask, trg_mask, src_pe, trg_pe)
            attns_dec_dec[:,i] = attn_dec_dec.detach()
            attns_dec_enc[:,i] = attn_dec_enc.detach()
        return self.norm(x), attns_dec_dec, attns_dec_enc


class LSTMDecoder(nn.Module):
    def __init__(self, hp):
        super(LSTMDecoder, self).__init__()
        self.hp = hp
        self.num_decoder_hidden_nodes = hp.num_hidden_nodes_decoder
        self.num_classes = hp.num_classes
        self.att = Attention(hp)
        # decoder
        self.L_sy = nn.Linear(self.num_decoder_hidden_nodes, self.num_decoder_hidden_nodes, bias=False)
        self.L_gy = nn.Linear(self.num_encoder_hidden_nodes, self.num_decoder_hidden_nodes)
        self.L_yy = nn.Linear(self.num_decoder_hidden_nodes, self.num_classes)

        self.L_ys = nn.Embedding(self.num_classes, self.num_decoder_hidden_nodes * 4)
        #self.L_ys = nn.Linear(self.num_classes, self.num_decoder_hidden_nodes * 4 , bias=False)
        self.L_ss = nn.Linear(self.num_decoder_hidden_nodes, self.num_decoder_hidden_nodes * 4, bias=False)
        self.L_gs = nn.Linear(self.num_encoder_hidden_nodes, self.num_decoder_hidden_nodes * 4)


    def forward(self, hbatch, lengths, targets):
        device = hbatch.device
        batch_size = hbatch.size(0)
        num_frames = hbatch.size(1)
        num_labels = targets.size(1)

        e_mask = torch.ones((batch_size, num_frames, 1), requires_grad=False).to(device, non_blocking=True)
        s = torch.zeros((batch_size, self.num_decoder_hidden_nodes), requires_grad=False).to(device, non_blocking=True)
        c = torch.zeros((batch_size, self.num_decoder_hidden_nodes), requires_grad=False).to(device, non_blocking=True)
        youtput = torch.zeros((batch_size, num_labels, self.num_classes), requires_grad=False).to(device, non_blocking=True)
        alpha = torch.zeros((batch_size, 1, num_frames), requires_grad=False).to(device, non_blocking=True)

        for i, tmp in enumerate(lengths):
            if tmp < num_frames:
                e_mask.data[i, tmp:] = 0.0

        for step in range(num_labels):
            g, alpha = self.att(s, hbatch, alpha, e_mask)
            # generate
            y = self.L_yy(torch.tanh(self.L_gy(g) + self.L_sy(s)))
            # recurrency calcuate
            rec_input = self.L_ys(targets[:, step]) + self.L_ss(s) + self.L_gs(g)
            s, c = self._func_lstm(rec_input, c)

            youtput[:, step] = y
        return youtput
    
    def decode_v2(self, hbatch, lengths, model_lm=None):
        """
        decode function with a few modification.
        1. Add the candidate when the prediction is </s>
        """
        device = hbatch.device
        #import sentencepiece as spm
        #sp = spm.SentencePieceProcessor()
        #sp.Load(self.hp.spm_model)
        batch_size = hbatch.shape[0]
        num_frames = hbatch.shape[1]
        e_mask = torch.ones((batch_size, num_frames, 1), device=device, requires_grad=False)

        beam_width = self.hp.beam_width
        beam_search = {'result': torch.zeros((beam_width, self.hp.max_decoder_seq_len), device=device, dtype=torch.long),
                       'length': torch.zeros(beam_width).long(),
                       'score': torch.zeros((beam_width), device=device, dtype=torch.float).fill_(0),
                       'c': torch.zeros((beam_width, self.num_decoder_hidden_nodes), device=device),
                       's': torch.zeros((beam_width, self.num_decoder_hidden_nodes), device=device),
                       'alpha': torch.zeros((beam_width, self.hp.max_decoder_seq_len, num_frames), device=device)}

        beam_results = {'score': torch.zeros((beam_width), device=device, dtype=torch.float).fill_(0),
                        'result': torch.zeros((beam_width, self.hp.max_decoder_seq_len), device=device, dtype=torch.long),
                        'length': torch.zeros(beam_width).long(),
                        'alpha': torch.zeros((beam_width, self.hp.max_decoder_seq_len, num_frames), device=device, requires_grad=False)}

        beam_step = 0

        for i, tmp in enumerate(lengths):
            if tmp < num_frames:
                e_mask[i, tmp:] = 0.0
        
        for seq_step in range(self.hp.max_decoder_seq_len):
            # length_penalty = ((5 + seq_step + 1)**0.9 / (5 + 1)**0.9)
            cand_seq = copy.deepcopy(beam_search['result'])
            cand_score = copy.deepcopy(beam_search['score'].unsqueeze(1))
            c = copy.deepcopy(beam_search['c'])
            s = copy.deepcopy(beam_search['s'])
            cand_alpha = copy.deepcopy(beam_search['alpha'])
            if seq_step == 0:
                g, alpha = self.att(s, hbatch, cand_alpha[:, seq_step, :].unsqueeze(1), e_mask)
            else:
                g, alpha = self.att(s, hbatch, cand_alpha[:, seq_step - 1, :].unsqueeze(1), e_mask)
            # generate
            y = self.L_yy(torch.tanh(self.L_gy(g) + self.L_sy(s)))

            if self.hp.score_func == 'log_softmax':
                y = F.log_softmax(y, dim=1)
                if model_lm is not None and seq_step > 0:
                    lm_input = cand_seq[:, :seq_step]
                    lm_score = model_lm(lm_input)[:, -1, :]
                    tmpy = y + self.hp.lm_weight * F.log_softmax(lm_score, dim=1)
                else:
                    tmpy = y.clone()
            elif self.hp.score_func == 'softmax':
                y = F.softmax(y, dim=1)
                if model_lm is not None and seq_step:
                    lm_input = cand_seq[:, :seq_step]
                    lm_score = model_lm(lm_input)[:, -1, :]
                    y = y + self.hp.lm_weight * F.softmax(lm_score, dim=1)
                else:
                    tmpy = y.clone()

            best_scores, best_indices = tmpy.data.topk(beam_width, dim=1)
            scores = cand_score + best_scores + 1 #0.5
            tmp_s = torch.zeros((beam_width, self.num_decoder_hidden_nodes), device=device)
            tmp_c = torch.zeros((beam_width, self.num_decoder_hidden_nodes), device=device)

            if seq_step == 0:
                beam_search['score'] = scores[0]
                beam_search['result'][:, 0] = best_indices[0]
                beam_search['length'] += 1
                beam_search['alpha'][:, 0, :] = alpha.squeeze(1)
                tmp_s = s
                tmp_c = c
                rec_input = self.L_ys(best_indices[0]) + self.L_ss(tmp_s) + self.L_gs(g)
                tmps, tmpc = self._func_lstm(rec_input, tmp_c)
                beam_search['s'] = tmps
                beam_search['c'] = tmpc
            else:
                k_scores, k_ix = scores.reshape(-1).topk(beam_width * 2)
                cand_idx = k_ix // beam_width
                cand_ids = k_ix % beam_width

                num_cand = 0
                i_cand = 0
                tmp_bestidx = torch.zeros((beam_width), dtype=torch.long, device=DEVICE)
                tmp_g = torch.zeros((beam_width, self.num_decoder_hidden_nodes * 2), dtype=torch.float, device=DEVICE)

                while num_cand < beam_width:
                    if best_indices[cand_idx[i_cand], cand_ids[i_cand]] == self.hp.eos_id:
                        beam_results['score'][beam_step] = k_scores[i_cand]
                        beam_results['result'][beam_step] = cand_seq[cand_idx[i_cand]]
                        beam_results['result'][beam_step][seq_step] = best_indices[cand_idx[i_cand], cand_ids[i_cand]]
                        beam_results['length'][beam_step] = seq_step + 1
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
                        beam_search['alpha'][num_cand] = cand_alpha[cand_idx[i_cand], :, :]
                        beam_search['alpha'][num_cand][seq_step] = alpha[cand_idx[i_cand]].squeeze(0)
                        tmp_s[num_cand] = s[cand_idx[i_cand]]
                        tmp_c[num_cand] = c[cand_idx[i_cand]]
                        tmp_g[num_cand] = g[cand_idx[i_cand]]

                        i_cand += 1
                        num_cand += 1

                    if beam_step >= beam_width:
                        break

                rec_input = self.L_ys(tmp_bestidx) + self.L_ss(tmp_s) + self.L_gs(tmp_g)
                tmps, tmpc = self._func_lstm(rec_input, tmp_c)
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
        #self._plot_attention(attention, results)

        return results
    
    @staticmethod
    def _func_lstm(x, c):
        ingate, forgetgate, cellgate, outgate = x.chunk(4, 1)
        half = 0.5
        ingate = torch.tanh(ingate * half) * half + half
        forgetgate = torch.tanh(forgetgate * half) * half + half
        cellgate = torch.tanh(cellgate)
        outgate = torch.tanh(outgate * half) * half + half
        c_next = (forgetgate * c) + (ingate * cellgate)
        h = outgate * torch.tanh(c_next)
        return h, c_next

    def _plot_attention(self, attention, label=None):
        import matplotlib.pyplot as plt
        import sentencepiece as spm
        attention = attention.cpu().numpy()
        sp = spm.SentencePieceProcessor()
        sp.Load(self.hp.spm_model)
        import pdb; pdb.set_trace()
        return 
        
