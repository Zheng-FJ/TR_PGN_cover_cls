''' This module will handle the text generation with beam search. '''

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer.Models import Transformer, get_pad_mask, get_subsequent_mask


class Translator(nn.Module):
    ''' Load a trained model and translate in beam search fashion. '''

    def __init__(
            self, model, beam_size, max_seq_len,
            src_pad_idx, trg_pad_idx, trg_bos_idx, trg_eos_idx, unk_idx,
            vocab_size, use_pointer):
        

        super(Translator, self).__init__()

        self.alpha = 0.7
        self.beam_size = beam_size
        self.max_seq_len = max_seq_len
        self.src_pad_idx = src_pad_idx
        self.trg_bos_idx = trg_bos_idx
        self.trg_eos_idx = trg_eos_idx
        self.unk_idx = unk_idx
        self.vocab_size = vocab_size

        self.model = model
        self.decoder_word_emb = model.decoder.trg_word_emb
        self.decoder_pos_emb = model.decoder.position_enc
        self.model.eval()

        self.register_buffer('init_seq', torch.LongTensor([[trg_bos_idx]]))
        self.register_buffer(
            'blank_seqs', 
            torch.full((beam_size, max_seq_len), trg_pad_idx, dtype=torch.long))
        self.blank_seqs[:, 0] = self.trg_bos_idx
        self.register_buffer(
            'len_map', 
            torch.arange(1, max_seq_len + 1, dtype=torch.long).unsqueeze(0))

        self.use_pointer = use_pointer
        self.use_cls_layers = model.use_cls_layers
        self.use_score_matrix = model.use_score_matrix
        self.q_based = model.q_based
        self.use_bce = model.use_bce
        self.use_rgere = model.use_regre
        self.utt_encode = model.utt_encode



    def _model_decode(self, trg_seq, enc_output, src_mask, score_matrix, src_seq_with_oov, oov_zero):
        trg_mask = get_subsequent_mask(trg_seq)

        dec_output, dec_slf_attn_list, dec_enc_attn_list, dec_hidden_states, enc_dec_contexts\
            = self.model.decoder(trg_seq, trg_mask, enc_output, src_mask, score_matrix, return_attns=self.use_pointer)
        vocab_dist = F.softmax(self.model.trg_word_prj(dec_output), dim=-1)
        if self.use_pointer:
            tgt_emb = self.decoder_pos_emb(self.decoder_word_emb(self.init_seq))
            p_gen = self.model.gen_prob(enc_dec_contexts, dec_hidden_states, tgt_emb)  # [b, tgt, 1]
            p_copy = (1 - p_gen)  # [b, tgt, 1]

            # [b, n, tgt, src] ---> [b, tgt, src]
            attention_score = dec_enc_attn_list[-1].sum(dim=1)
            attention_score = F.softmax(attention_score, dim=-1)
            vocab_dist_p = vocab_dist * p_gen
            context_dist_p = attention_score * p_copy


            if oov_zero is not None:
                oov_zore_extend = oov_zero.unsqueeze(1).repeat([vocab_dist_p.size(0), attention_score.size(1), 1])
                vocab_dist_p = torch.cat([vocab_dist_p, oov_zore_extend], dim=-1)

            src_seq_with_oov_extend = src_seq_with_oov.unsqueeze(1).repeat([vocab_dist_p.size(0), attention_score.size(1), 1])
            final_dist = vocab_dist_p.scatter_add(dim=-1, index=src_seq_with_oov_extend, src=context_dist_p)
        else:
            final_dist = vocab_dist


        return final_dist

    def _utts_process(self, input_repre, sections, utt_num, padding = False, utt_merge=False):
        b_s = input_repre.shape[0]
        size = input_repre.shape[1]
        emb_dim = input_repre.shape[2]
        total_utt_num = torch.sum(utt_num)
        max_utt_num = torch.max(utt_num)
        res = torch.zeros([total_utt_num, emb_dim], dtype = torch.float32).cuda()
        res1 = torch.zeros([b_s, max_utt_num, emb_dim], dtype=torch.float32).cuda() 

        if utt_merge:
            idx2 = 0
            for i in range(b_s):
                for utt in input_repre[i, :utt_num[i], :]:
                    res[idx2] += utt
                    idx2 += 1

            return res


        idx = 0
        for i in range(b_s):
            section = sections[i].cpu().numpy().tolist()
            section = [x for x in section if x != 0]
            # print(len(section))
            if size > sum(section):
                section.append(size-sum(section))

            splited_utt = torch.split(input_repre[i], section, dim=0)
            splited_utt = splited_utt[:-1]

            if padding:        
                idx1 = 0
                for utt in splited_utt:
                    res1[i, idx1] += utt[0, :]
                    idx1 += 1
            else:

                for utt in splited_utt:
                    res[idx] += (utt[0, :])
                    idx += 1

        if padding:
            res1_mask = get_pad_mask(res1[:,:,0], pad_idx=0.0)
            return res1, res1_mask
        else:
            return res

    def _get_init_state(self, src_seq, src_mask, src_seq_with_oov, oov_zero, attn_mask1, attn_mask2, attn_mask3, article_lens, utt_num):
        beam_size = self.beam_size
        enc_output, enc_slf_attn_list, enc_hidden_states \
            = self.model.encoder(src_seq, src_mask, return_attns=self.use_pointer, attn_mask1=attn_mask1, attn_mask2=attn_mask2, attn_mask3=attn_mask3)

        if self.use_cls_layers and article_lens is not None and utt_num is not None:
            # enc_utt_output = self._utts_process(enc_output, article_lens, utt_num)
            ''' utterance encode '''
            if self.utt_encode:
                enc_utt_output, enc_utt_mask = self._utts_process(enc_output, article_lens, utt_num, padding = True, utt_merge=False)
                enc_utt_output = self.model.utt_encoder(enc_utt_output, src_mask=enc_utt_mask, return_attns=False, attn_mask1=None, attn_mask2=None, attn_mask3=None)
                enc_utt_output = self._utts_process(enc_utt_output, article_lens, utt_num, padding=False, utt_merge=True)
            else:
                enc_utt_output  = self._utts_process(enc_output, article_lens, utt_num, padding=False, utt_merge=False)

            ''' Q_based'''
            if self.q_based:
                enc_utt_output = torch.split(enc_utt_output,utt_num.cpu().numpy().tolist())
                enc_utt_output = list(enc_utt_output)
                for i in range(len(enc_utt_output)):
                    q_repre = enc_utt_output[i][0].unsqueeze(0).repeat([enc_utt_output[i].shape[0], 1])
                    enc_utt_output[i] = torch.cat((q_repre, enc_utt_output[i]), dim=-1)
                enc_utt_output = tuple(enc_utt_output)
                enc_utt_output = torch.cat(enc_utt_output, dim = 0)


            output_logit = self.model.classify_layer(enc_utt_output)
            if self.use_bce:
                # predicted_label = torch.argmax(output_logit, dim = -1)
                predicted_label = output_logit.squeeze(1)
            elif self.use_rgere:
                predicted_label = output_logit.squeeze(1)

            ''' score_matrix '''
            if self.use_score_matrix:
                utts_score = torch.zeros([output_logit.shape[0] + 1], dtype = torch.float32).cuda()
                if self.use_bce:
                    utts_score[:-1] += output_logit[:,0]
                elif self.use_rgere:
                    utts_score[:-1] += output_logit[:,0]

                max_n_words = enc_output.shape[1]
                utt_idx = 0
                prob_indices = []

                for nums_words in article_lens:
                    indices = []
                    # print(nums_words)
                    for num_words in nums_words:
                        if num_words == 0:
                            # print('0 words')
                            continue
                        indices += [utt_idx] * num_words
                        utt_idx += 1
                    indices += [-1] * (max_n_words - sum(nums_words))
                    prob_indices.append(indices)
                prob_indices = torch.tensor(prob_indices)   #[batch, 300] ([batch, max_input_len])

                '''构造分数矩阵'''
                score_matrix = utts_score[prob_indices].unsqueeze(1).unsqueeze(1).repeat([1, 8, 50, 1]) # 这里默认最大target长度就是50
                score_matrix = torch.clamp(score_matrix, min = 0.2, max = 0.8)  # 数据平滑，防止分化太严重
            else:
                score_matrix =None

        else:
            predicted_label = None
            score_matrix = None
        
        final_dist = self._model_decode(self.init_seq, enc_output, src_mask, score_matrix, src_seq_with_oov, oov_zero)

        best_k_probs, best_k_idx = final_dist[:, -1, :].topk(beam_size)

        scores = torch.log(best_k_probs).view(beam_size)

        gen_seq = self.blank_seqs.clone().detach()
        gen_seq[:, 1] = best_k_idx[0]
        enc_output = enc_output.repeat(beam_size, 1, 1)


        return enc_output, gen_seq, scores, predicted_label, score_matrix



    def _get_the_best_score_and_idx(self, gen_seq, dec_output, scores, step):
        assert len(scores.size()) == 1
        
        beam_size = self.beam_size

        # Get k candidates for each beam, k^2 candidates in total.
        best_k2_probs, best_k2_idx = dec_output[:, -1, :].topk(beam_size)

        # Include the previous scores.
        scores = torch.log(best_k2_probs).view(beam_size, -1) + scores.view(beam_size, 1)

        # Get the best k candidates from k^2 candidates.
        scores, best_k_idx_in_k2 = scores.view(-1).topk(beam_size)
 
        # Get the corresponding positions of the best k candidiates.
        best_k_r_idxs, best_k_c_idxs = best_k_idx_in_k2 // beam_size, best_k_idx_in_k2 % beam_size
        best_k_idx = best_k2_idx[best_k_r_idxs, best_k_c_idxs]

        # Copy the corresponding previous tokens.
        gen_seq[:, :step] = gen_seq[best_k_r_idxs, :step]
        # Set the best tokens in this beam search step
        gen_seq[:, step] = best_k_idx

        return gen_seq, scores


    def translate_sentence(self, src_seq, src_seq_with_oov_extend, oov_zero, attn_mask1, attn_mask2, attn_mask3, article_lens, utt_num):
        # Only accept batch size equals to 1 in this function.
        # TODO: expand to batch operation.
        assert src_seq.size(0) == 1

        src_pad_idx = self.src_pad_idx
        trg_eos_idx = self.trg_eos_idx
        max_seq_len = self.max_seq_len
        beam_size = self.beam_size
        alpha = self.alpha

        with torch.no_grad():
            src_mask = get_pad_mask(src_seq, src_pad_idx)
            enc_output, gen_seq, scores, predicted_label, score_matrix = self._get_init_state(src_seq, src_mask, src_seq_with_oov_extend, oov_zero, attn_mask1=attn_mask1, attn_mask2=attn_mask2, attn_mask3=attn_mask3, article_lens=article_lens, utt_num=utt_num)
            #print("init_scores: ",scores)

            ans_idx = 0   # default
            for step in range(2, max_seq_len):    # decode up to max length
                # print(gen_seq[:, :step])
                # exit()
                current_tokens_idx = gen_seq.cpu().numpy().tolist()
                current_tokens_idx = [
                    [token if token < self.vocab_size else self.unk_idx for token in line]
                    for line in current_tokens_idx]
                current_tokens_idx = torch.tensor(current_tokens_idx, dtype=torch.long).cuda()
                final_dist = self._model_decode(current_tokens_idx[:, :step], enc_output, src_mask, score_matrix, src_seq_with_oov_extend, oov_zero)

                gen_seq, scores = self._get_the_best_score_and_idx(gen_seq, final_dist, scores, step)
                #print("scores: ",scores)

                # Check if all path finished
                # -- locate the eos in the generated sequences
                eos_locs = gen_seq == trg_eos_idx   
                # -- replace the eos with its position for the length penalty use
                seq_lens, _ = self.len_map.masked_fill(~eos_locs, max_seq_len).min(1)
                # -- check if all beams contain eos
                if (eos_locs.sum(1) > 0).sum(0).item() == beam_size:
                    # TODO: Try different terminate conditions.
                    _, ans_idx = scores.div(seq_lens.float() ** alpha).max(0)
                    ans_idx = ans_idx.item()
                    break
        return gen_seq[ans_idx][:seq_lens[ans_idx]].tolist(), predicted_label
