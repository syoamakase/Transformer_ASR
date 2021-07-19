#-*- coding: utf-8 -*-

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
import random

from utils.utils import npeak_mask, frame_stacking

class CTCWav2vec2(nn.Module):
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
        self.hp = hp
        self.trg_vocab = hp.vocab_size

        self.decoder = nn.Linear(1024, self.trg_vocab)

    def forward(self, src):

        ctc_outputs = self.decoder(src)
        outputs, attn_dec_dec, attn_dec_enc = None, None, None
        return outputs, ctc_outputs, attn_dec_dec, attn_dec_enc

    @torch.no_grad()
    def decode(self, src):
        decoder_output = self.decoder(src)
        batch_size = decoder_output.shape[0]
        results = []
        prev_id = self.trg_vocab + 1
        for b in range(batch_size):
            results_batch = []
            for x in decoder_output[b].argmax(dim=1):
                if int(x) != prev_id and int(x) != 0:
                    results_batch.append(int(x))
                prev_id = int(x)
            results.append(results_batch)
        # TODO: implement batch
        results = results[0]
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
        #if hasattr(self, 'linear'):
        #    e_outputs = self.linear(e_outputs)
    
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


    def _freq_mask(self, spec, F=10, num_masks=1, replace_with_zero=False, random_mask=False, granularity=1):
        cloned = spec.clone()
        num_mel_channels = cloned.shape[2]
        for j in range(cloned.shape[0]):
            for i in range(0, num_masks):
                f = random.randrange(0, F, granularity)
                if random_mask:
                    sample = np.arange(0, num_mel_channels)
                    masks = random.sample(list(sample), f)
                    if (replace_with_zero): cloned[j, :, masks] = 0
                    else: cloned[j, :, masks] = cloned.mean()
                else:
                    f_zero = random.randrange(0, num_mel_channels - f)
                    # avoids randrange error if values are equal and range is empty
                    if (f_zero == f_zero + f*granularity): return cloned
                    mask_end = random.randrange(f_zero, f_zero + f, granularity)
                    if (replace_with_zero): cloned[j, :, f_zero:mask_end] = 0
                    else: cloned[j, :, f_zero:mask_end] = cloned.mean()
        return cloned
    
    def _time_mask(self, spec, T=50, num_masks=1, replace_with_zero=False, random_mask=False):
        cloned = spec.clone()
        len_spectro = cloned.shape[1]

        for j in range(cloned.shape[0]):
            for i in range(0, num_masks):
                t = random.randrange(0, T)
                t_zero = random.randrange(0, len_spectro - t)
    
                # avoids randrange error if values are equal and range is empty
                if (t_zero == t_zero + t): return cloned
    
                mask_end = random.randrange(t_zero, t_zero + t)
                if (replace_with_zero): cloned[j, t_zero:mask_end,:] = 0
                else: cloned[j, t_zero:mask_end,:] = cloned.mean()
        return cloned

    def spec_aug(self, features):
        feature_length = features.shape[1]
        num_T = min(20, math.floor(0.04*feature_length))
        T = math.floor(0.04*feature_length)

        features = self._time_mask(self._freq_mask(features, F=self.hp.spec_size_f, num_masks=self.hp.num_F, replace_with_zero=True, random_mask=self.hp.random_mask, granularity=self.hp.granularity), T=T, num_masks=num_T, replace_with_zero=True)
        return features
