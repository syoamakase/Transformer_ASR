#-*- coding: utf-8 -*-

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
import random

from Models.modules import CNN_embedding, CNN_embedding_avepool
from Models.encoder import Encoder, ConformerEncoder
from Models.decoder import Decoder, LSTMDecoder
from transformers import Wav2Vec2Model
from utils.utils import npeak_mask, frame_stacking


class TransformerWav2vec2(nn.Module):
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
    def __init__(self, hp, pretrain_model, freeze_feature_extractor=False):
        super().__init__()
        self.hp = hp
        self.d_model_d = hp.d_model_d
        self.trg_vocab = hp.vocab_size
        self.encoder_type = hp.encoder
        self.decoder_type = hp.decoder
        self.use_ctc = hp.use_ctc
        self.freeze_feature_extractor = freeze_feature_extractor
        self.iter_freeze_encoder = hp.iter_freeze_encoder

        if self.decoder_type == 'ctc' and self.use_ctc:
            warnings.warn(f"hp.decoder == 'ctc' and hp.use_ctc is True, hp.use_ctc is changed to False")
            self.use_ctc = False
        
        self.frame_stacking = True if hp.frame_stacking is not None else False

        self.encoder = Wav2Vec2Model.from_pretrained(pretrain_model)

        #self.encoder.config.mask_feature_prob = hp.feature_mask

        if self.freeze_feature_extractor:
            print('freeze parameters')
            self.encoder.feature_extractor._freeze_parameters()

        if self.encoder_type.lower() == 'conformer':
            if hp.cnn_avepool:
                self.cnn_encoder = CNN_embedding_avepool(hp)
            else:
                self.cnn_encoder = CNN_embedding(hp)
                self.encoder_asr = ConformerEncoder(hp)
                self.decoder = LSTMDecoder(hp)
        else:
            self.dropout = nn.Dropout(0.1)
            if self.decoder_type.lower() == 'transformer':
                self.linear = nn.Linear(1024, self.d_model_d)
                self.decoder = Decoder(hp)
                self.out = nn.Linear(self.d_model_d, self.trg_vocab)
            elif self.decoder_type.lower() == 'ctc':
                self.decoder = nn.Linear(1024, self.trg_vocab)
            else:
                self.linear = nn.Linear(1024, self.d_model_d)
                self.decoder = LSTMDecoder(hp)

        if self.use_ctc:
            self.out_ctc = nn.Linear(1024, self.trg_vocab)

    def freeze_feature_extractor(self):
        self.encoder.feature_extractor._freeze_parameters()

    def forward(self, src, trg, src_mask, trg_mask, num_update, attention_mask=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                labels=None,
                update=True
        ):

        if self.freeze_feature_extractor:
            if self.iter_freeze_encoder > num_update:
                with torch.no_grad():
                    print('NOT update because of freeze_encoder')
                    outputs = self.encoder(src, attention_mask=attention_mask, output_attentions=output_attentions,
                                           output_hidden_states=output_hidden_states, return_dict=return_dict)
            else:
                if self.iter_freeze_encoder == num_update:
                    print('wav2vec2 encoder update')
                if update:
                    print('update')
                    outputs = self.encoder(src, attention_mask=attention_mask, output_attentions=output_attentions,
                                          output_hidden_states=output_hidden_states, return_dict=return_dict)
                else:
                    print('NOT update')
                    with torch.no_grad():
                        outputs = self.encoder(src, attention_mask=attention_mask, output_attentions=output_attentions,
                                               output_hidden_states=output_hidden_states, return_dict=return_dict)

        else:
            print('NOT update freeze')
            with torch.no_grad():
                outputs = self.encoder(src, attention_mask=attention_mask, output_attentions=output_attentions,
                                       output_hidden_states=output_hidden_states, return_dict=return_dict)

        hidden_states = outputs[0]

        if self.encoder_type.lower() == 'conformer':
            ## feature mask
            if num_update > self.hp.warmup_step:
                hidden_states = self.spec_aug(hidden_states)
                
            src, src_mask = self.cnn_encoder(hidden_states, src_mask)

            e_outputs, attn_enc_enc = self.encoder_asr(src, src_mask)
            d_output, attn_dec_dec, attn_dec_enc = self.decoder(trg, e_outputs, src_mask, trg_mask)
            outputs = d_output
            if self.use_ctc:
                ctc_outputs = self.out_ctc(hidden_states)
            else:
                ctc_outputs = None
        else:
            hidden_states = self.dropout(hidden_states)
            if self.decoder_type.lower() == 'ctc':
                ctc_outputs = self.decoder(hidden_states)
                outputs, attn_dec_dec, attn_dec_enc = None, None, None
            else:
                e_outputs = self.linear(hidden_states)
                d_output, attn_dec_dec, attn_dec_enc = self.decoder(trg, e_outputs, src_mask, trg_mask)
                if self.decoder_type.lower() == 'transformer':
                    outputs = self.out(d_output)
                else: # 'LSTM'
                    outputs = d_output

                if self.use_ctc:
                    ctc_outputs = self.out_ctc(hidden_states)
                else:
                    ctc_outputs = None
        return outputs, ctc_outputs, attn_dec_dec, attn_dec_enc

    @torch.no_grad()
    def decode(self, src, src_dummy, beam_size=10, model_lm=None, init_tok=2, eos_tok=1, lm_weight=0.2,
               attention_mask=None,
               output_attentions=None,
               output_hidden_states=None,
               return_dict=None,
               labels=None,
               ):
        with torch.no_grad():
            outputs = self.encoder(src, attention_mask=attention_mask, output_attentions=output_attentions,
                                   output_hidden_states=output_hidden_states, return_dict=return_dict)

            src_mask = (src_dummy != 0).unsqueeze(-2)
            hidden_states = outputs[0]
            if self.encoder_type.lower() == 'conformer':
                ## feature mask
                src, src_mask = self.cnn_encoder(hidden_states, src_mask)

                e_outputs, attn_enc_enc = self.encoder_asr(src, src_mask)
                ## TODO: lm
                results = self.decoder.decode_v2(e_outputs, src_mask, model_lm, lm_weight)
                if self.use_ctc:
                    ctc_outputs = self.out_ctc(hidden_states)
                else:
                    ctc_outputs = None
            elif self.decoder_type.lower() == 'transformer':
                e_outputs = self.linear(hidden_states)
                results = self._decode_tranformer_decoder(e_outputs, src_mask, beam_size, model_lm, init_tok, eos_tok, lm_weight)
            elif self.decoder_type.lower() == 'ctc':
                hidden_states = self.dropout(hidden_states)
                decoder_output = self.decoder(hidden_states)
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
            else:
                hidden_states = self.dropout(hidden_states)
                e_outputs = self.linear(hidden_states)
                results = self.decoder.decode_v2(e_outputs, src_mask, model_lm, lm_weight)
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
