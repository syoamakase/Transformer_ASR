#-*- coding: utf-8 -*-

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.modules import CNN_embedding, CNN_embedding_avepool
from Models.encoder import Encoder, ConformerEncoder
from Models.decoder import Decoder, LSTMDecoder
from utils.utils import npeak_mask, frame_stacking

class Transformer(nn.Module):
    """
    Transformer ASR model.
    It means the encoder uses Transformer (Conformer) and the decoder also uses Transformer.
    Args (from hparams.py):
        d_model_e (int): model dimension of encoder
        d_model_d (int): model dimension of decoder
        vocab_size (int): target vocabulary size
        encoder (str): encoder architecture (transformer or conformer)
        decoder (str): decoder architecture (transformer or LSTM (dev))
        mode (str): outputs type (transformer or ctc-transformer)
        frame_stacking (int): If 1 (NOT using frame stacking), it uses CNN subsampling
    """
    def __init__(self, hp):
        super().__init__()
        self.d_model_e = hp.d_model_e
        self.d_model_d = hp.d_model_d
        self.trg_vocab = hp.vocab_size
        self.encoder_type = hp.encoder
        self.decoder_type = hp.decoder
        self.mode = hp.mode
        self.frame_stacking = True if hp.frame_stacking > 1 else False

        if not self.frame_stacking:
            if hp.cnn_avepool:
                self.cnn_encoder = CNN_embedding_avepool(hp)
            else:
                self.cnn_encoder = CNN_embedding(hp)
            self.cnn_encoder = CNN_embedding(hp)
        else:
            self.embedder = nn.Linear(hp.mel_dim*hp.frame_stacking, self.d_model_e)

        if self.encoder_type == 'Conformer':
            self.encoder = ConformerEncoder(hp)
        else:
            self.encoder = Encoder(hp)
        if self.decoder_type.lower() == 'transformer':
            self.decoder = Decoder(hp)
            self.out = nn.Linear(self.d_model_d, self.trg_vocab)
        else:
            self.decoder = LSTMDecoder(hp)

        if self.mode == 'ctc-transformer':
            self.out_ctc = nn.Linear(self.d_model_e, self.trg_vocab)

    def forward(self, src, trg, src_mask, trg_mask):
        if not self.frame_stacking:
            src, src_mask = self.cnn_encoder(src, src_mask)
        else:
            src = self.embedder(src)

        e_outputs, attn_enc_enc = self.encoder(src, src_mask)
        d_output, attn_dec_dec, attn_dec_enc = self.decoder(trg, e_outputs, src_mask, trg_mask)
        if self.decoder_type.lower() == 'transformer':
            outputs = self.out(d_output)
        else:
            outputs = d_output

        if self.mode == 'ctc-transformer':
            ctc_outputs = self.out_ctc(e_outputs)
        else:
            ctc_outputs = None
        return outputs, ctc_outputs, attn_enc_enc, attn_dec_dec, attn_dec_enc

    @torch.no_grad()
    def decode(self, src, src_dummy, beam_size=10, model_lm=None, init_tok=2, eos_tok=1, lm_weight=0.0):
        with torch.no_grad():
            if not self.frame_stacking:
                src_mask = (src_dummy != 0).unsqueeze(-2)
                src, src_mask = self.cnn_encoder(src, src_mask)
            else:
                src_mask = (src_dummy != 0)
                src, src_mask = frame_stacking(src, src_mask, 3)
                src = self.embedder(src)

            e_output, _ = self.encoder(src, src_mask)

            if self.decoder_type.lower() == 'transformer':
                results = self._decode_tranformer_decoder(e_output, src_mask, beam_size, model_lm, init_tok, eos_tok, lm_weight)
            else:
                results = self.decoder.decode_v2(e_output, src_mask, model_lm)
                # decode_v2(self, hbatch, lengths, model_lm=None):

        return results

    def _decode_tranformer_decoder(self, e_output, src_mask, beam_size, model_lm, init_tok, eos_tok, lm_weight):
        """
        Decoding for tranformer decoder
        Args:
            e_output (torch.Float.tensor): Encoder outputs (B x T x d_model_e)
            src_mask (torch.Bool.tensor): If False at the `t`, a time step of t is padding value.
            beam_width (int, optional): Beam size. Default: 1
            model_lm (torch.nn.Module, optional): Language model for shallow fusion. If None, shallow fusion is disabled. Default: None
            init_tok (int, optional): ID of <sos> token. Default: 2
            eos_tok (int, optional): ID of <eos> token. Default: 1
            lm_weight (float, optional): Weight of lauguage model in shallow fusion. If 0.0, it is equivalent to disabling shallow fusion


        Returns:
            list: The best result of beam search.
            int: Lengths of the best result.
        """
        max_len = 300
        if hasattr(self, 'linear'):
            e_outputs = self.linear(e_outputs)
    
        device = e_output.device
        outputs = torch.LongTensor([[init_tok]])
        trg_mask = npeak_mask(1)

        out = self.out(self.decoder(outputs.to(device), e_output, src_mask, trg_mask)[0])

        out = F.softmax(out, dim=-1)
        probs, ix = out[:, -1].data.topk(beam_size)
        log_scores = torch.Tensor([torch.log(prob) for prob in probs.data[0]]).unsqueeze(0)

        outputs = torch.zeros((beam_size, max_len), device=device).long()
        outputs[:, 0] = init_tok
        outputs[:, 1] = ix[0]
        e_outputs = torch.zeros((beam_size, e_output.size(-2), e_output.size(-1)), device=device)
        e_outputs[:, :] = e_output[0]

        beam_results = {'score': torch.zeros((beam_size),device=device, dtype=torch.float).fill_(-100),
                        'result':torch.zeros((beam_size, max_len), device=device, dtype=torch.long),
                        'length':torch.zeros(beam_size).long()}

        beam_step = 0
        end_beam = False
        for i in range(2, max_len):
            trg_mask = npeak_mask(i)
            batch_size = outputs.shape[0]
            e_outputs_ = e_outputs[:batch_size]
            src_mask_ = src_mask[:batch_size]

            out = self.out(self.decoder(outputs[:, :i].to(device), e_outputs_, src_mask_, trg_mask)[0])

            asr_score = F.log_softmax(out[:, -1], dim=1).data
            if model_lm is not None:
                lm_score = F.log_softmax(model_lm(outputs[:, :i]), dim=2)[:, -1]
            else:
                lm_score = 0

            # Length penalty. Please see [Wu+, 2016] https://arxiv.org/abs/1609.08144
            # In this setting, alpha = 0.9
            lengths_penalty = ((5+i)**0.9 / (5+1)**0.9)

            total_score = asr_score + 0.2*lm_score + lengths_penalty

            probs, ix = total_score.data.topk(batch_size)

            log_probs = torch.Tensor([p.cpu() for p in probs.data.view(-1)]).view(batch_size, -1) + log_scores.transpose(0,1)
            k_probs, k_ix = log_probs.reshape(-1).topk(beam_size)
            row = k_ix // batch_size
            col = k_ix % batch_size
            outputs_new = torch.zeros((beam_size, i+1)).long().to(device)
            outputs_new[:, :i] = outputs[row, :i]
            outputs_new[:, i] = ix[row, col]
            log_scores = k_probs.unsqueeze(0)
            outputs = copy.deepcopy(outputs_new)

            outputs_new = []
            log_scores_new = []
            for kk in range(beam_size):
                if outputs[kk, i] == eos_tok:
                    beam_results['score'][beam_step] = copy.deepcopy(log_scores[0][kk])
                    beam_results['result'][beam_step, :i+1] = copy.deepcopy(outputs[kk,:i+1])
                    beam_results['length'][beam_step] = i+1
                    beam_step += 1
                    if kk == 0:
                        end_beam = True
                        break

                    if beam_step == beam_size:
                        end_beam = True
                        break

                else:
                    outputs_new.append(outputs[kk,:i+2].cpu().numpy())
                    log_scores_new.append(log_scores[0][kk].item())

            outputs = torch.tensor(outputs_new).long().to(device)
            log_scores = torch.tensor(log_scores_new).float().unsqueeze(0)

            if beam_step == 20 or end_beam or outputs.shape[0] < 5:
                break

        bestidx = beam_results['score'].argmax()
        length = beam_results['length'][bestidx]
        return beam_results['result'][bestidx,:length].cpu().tolist()
