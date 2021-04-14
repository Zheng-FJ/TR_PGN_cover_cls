''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from transformer.Layers import EncoderLayer, DecoderLayer


__author__ = "Yu-Hsiang Huang"


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy
        self.d_hid = d_hid
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x*np.sqrt(self.d_hid) + self.pos_table[:, :x.size(1)].clone().detach()


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=200):

        super().__init__()

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, src_seq, src_mask, return_attns=False, attn_mask1 = None, attn_mask2 = None, attn_mask3 = None):

        enc_slf_attn_list = []

        # -- Forward
        
        enc_output = self.dropout(self.position_enc(self.src_word_emb(src_seq)))
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn, enc_hidden_states = enc_layer(enc_output, slf_attn_mask=src_mask, attn_mask1 = attn_mask1, attn_mask2 = attn_mask2, attn_mask3 = attn_mask3)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []
        #print(len(enc_slf_attn_list))

        if return_attns:
            return enc_output, enc_slf_attn_list, enc_hidden_states
        return enc_output


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, n_trg_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, n_position=200, dropout=0.1):

        super().__init__()


        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False, cover=None):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Forward
        dec_output = self.dropout(self.position_enc(self.trg_word_emb(trg_seq)))

        dec_output = self.layer_norm(dec_output)
        # print(dec_output)
        # print(dec_output.size())
        # print(enc_output)
        # print(enc_output.size())
        # print(src_mask)
        # print(src_mask.size())
        # print(trg_mask)
        # print(trg_mask.size())
        # exit()

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn, dec_hidden_states, enc_dec_contexts = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask, cover=cover)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []
        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list, dec_hidden_states, enc_dec_contexts
        return dec_output

class GeneraProb(nn.Module):
    def __init__(self, d_model, dropout):
        super(GeneraProb, self).__init__()

        self.w_h = nn.Linear(d_model, 1)
        self.w_s = nn.Linear(d_model, 1)
        self.w_x = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(p=dropout)

    # h : weight sum of encoder output ,(batch,hidden*2)
    # s : decoder state                 (batch,hidden*2)
    # x : decoder input                 (batch,embed)
    def forward(self, h, s, x):
        h_feature = self.dropout(self.w_h(h))  # (batch,1)
        s_feature = self.dropout(self.w_s(s))  # (batch,1)
        x_feature = self.dropout(self.w_x(x))  # (batch,1)

        gen_feature = h_feature + s_feature + x_feature  # (batch,1)

        gen_p = torch.sigmoid(gen_feature)

        return gen_p

class label_classification(nn.Module):
    ''' label classify for encoder output'''
    def __init__(self, n_feature,  n_hidden1, n_hidden2, n_output):
        super().__init__()
        '''
        n_feature: 编码器输出embedding维度
        '''

        self.hidden_layer1 = nn.Linear(n_feature, n_hidden1)
        self.hidden_layer2 = nn.Linear(n_hidden1, n_hidden2)
        self.classify_output = nn.Linear(n_hidden2, n_output)
    
    def forward(self, encoder_output):

        output = torch.sigmoid(self.hidden_layer1(encoder_output))
        output=  torch.sigmoid(self.hidden_layer2(output))
        output = self.classify_output(output)

        output_logit = F.softmax(output, dim = -1)      

        return output_logit

def split_utts(input_repre, sections, max_art_len):
    b_s = input_repre.shape[0]
    size = input_repre.shape[1]
    emb_dim = input_repre.shape[2]
    max_utt_num = max_art_len[0]
    # print(max_art_len[0])
    # exit()
    res = torch.zeros([b_s, max_utt_num, emb_dim], dtype=torch.float).cuda()
    for i in range(b_s):
        section = sections[i].cpu().numpy().tolist()
        section = [x for x in section if x != 0]
        if size > sum(section):
            section.append(size-sum(section))

        splited_utt = torch.split(input_repre[i], section, dim=0)
        # 接下来做平均，把结果放到元组里
        tmp = torch.zeros([max_utt_num, emb_dim], dtype=torch.float).cuda() 
        # print(tmp.shape)
        # print(len(splited_utt))

        for j in range(len(splited_utt)):
            if j == max_utt_num:
                break
            utt = torch.mean(splited_utt[j], dim = 0)
            # if torch.isnan(utt.sum()):
            #     print(splited_utt[j])
            #     print(utt)
            #     exit()
            tmp[j] += utt


        res[i] += tmp


    return res

def utts_process(input_repre, sections, utt_num):
    b_s = input_repre.shape[0]
    size = input_repre.shape[1]
    emb_dim = input_repre.shape[2]
    total_utt_num = torch.sum(utt_num)
    res = torch.zeros([total_utt_num, emb_dim], dtype = torch.float32).cuda()

    idx = 0
    for i in range(b_s):
        section = sections[i].cpu().numpy().tolist()
        section = [x for x in section if x != 0]
        # print(len(section))
        if size > sum(section):
            section.append(size-sum(section))

        splited_utt = torch.split(input_repre[i], section, dim=0)
        splited_utt = splited_utt[:-1]
        for utt in splited_utt:
            res[idx] += (utt[0, :])
            idx += 1
    # print(total_utt_num)
    # print(res.shape)
    # exit()
    return res


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self, n_src_vocab, n_trg_vocab, src_pad_idx, trg_pad_idx,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200,
            trg_emb_prj_weight_sharing=True, emb_src_trg_weight_sharing=True,
            use_pointer=False, use_cls_layers=False):

        super().__init__()

        self.use_pointer = use_pointer

        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, dropout=dropout)

        self.decoder = Decoder(
            n_trg_vocab=n_trg_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=trg_pad_idx, dropout=dropout)

        self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        self.x_logit_scale = 1.
        if trg_emb_prj_weight_sharing:
            # Share the weight between target word embedding & last dense layer
            self.trg_word_prj.weight = self.decoder.trg_word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)

        if emb_src_trg_weight_sharing:
            self.encoder.src_word_emb.weight = self.decoder.trg_word_emb.weight

        self.gen_prob = GeneraProb(d_model, dropout)

        self.use_cls_layers = use_cls_layers
        if self.use_cls_layers:
            self.classify_layer = label_classification(n_feature = d_model, n_hidden1 = 256, n_hidden2 = 128, n_output = 2)

    def forward(self, src_seq, trg_seq, src_seq_with_oov, oov_zero, attn_mask1, attn_mask2, attn_mask3, cover, article_lens, utt_num):

        src_mask = get_pad_mask(src_seq, self.src_pad_idx)
        trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)

        
        if self.use_pointer:
            enc_output, enc_slf_attn_list, enc_hidden_states = self.encoder(src_seq, src_mask, return_attns=self.use_pointer, attn_mask1=attn_mask1, attn_mask2=attn_mask2, attn_mask3=attn_mask3)
            # # 这里词的表示要总结成句子表示
            # enc_utt_output = split_utts(enc_output, article_lens, max_art_len)
            if self.use_cls_layers:
                '''这里要取出每个句子的第一个表示，然后把一个batch里面的句子结合在一起'''
                enc_utt_output = utts_process(enc_output, article_lens, utt_num)

                # 在这里加线性层，返回二维Logit
                output_logit = self.classify_layer(enc_utt_output)
            else:
                output_logit = None
            
            dec_output, dec_slf_attn_list, dec_enc_attn_list, dec_hidden_states, enc_dec_contexts\
                = self.decoder(trg_seq, trg_mask, enc_output, src_mask, return_attns=self.use_pointer, cover=cover)
            tgt_emb = self.decoder.position_enc(self.decoder.trg_word_emb(trg_seq))
            p_gen = self.gen_prob(enc_dec_contexts, dec_hidden_states, tgt_emb) #[b, tgt, 1]
            p_copy = (1 - p_gen) #[b, tgt, 1]

        else:
            enc_output = self.encoder(src_seq, src_mask, attn_mask1=attn_mask1, attn_mask2=attn_mask2, attn_mask3=attn_mask3)
            dec_output = self.decoder(trg_seq, trg_mask, enc_output, src_mask, cover=cover)

        vocab_logit = self.trg_word_prj(dec_output) * self.x_logit_scale
        vocab_dist = torch.softmax(vocab_logit, dim=-1)

        if self.use_pointer:
            # [b, n, tgt, src] ---> [b, tgt, src]
            attention_score = dec_enc_attn_list[-1].sum(dim=1)
            attention_score = torch.softmax(attention_score, dim=-1)
            vocab_dist_p = vocab_dist * p_gen
            context_dist_p = attention_score * p_copy


            if oov_zero is not None:
                oov_zore_extend = oov_zero.unsqueeze(1).repeat([1, attention_score.size(1), 1])
                vocab_dist_p = torch.cat([vocab_dist_p, oov_zore_extend], dim=-1)

            src_seq_with_oov_extend = src_seq_with_oov.unsqueeze(1).repeat([1, attention_score.size(1), 1])
            final_dist = vocab_dist_p.scatter_add(dim=-1, index=src_seq_with_oov_extend, src=context_dist_p)
        else:
            final_dist = vocab_dist
            


        return final_dist, output_logit #[b, tgt, vocab]
        # return seq_logit.view(-1, seq_logit.size(2))
