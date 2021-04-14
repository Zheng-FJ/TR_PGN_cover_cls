''' Define the Layers '''
import torch.nn as nn
import torch
from transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward, MultiHeadAttention_with_extra_mask


__author__ = "Yu-Hsiang Huang"


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.slf_attn = MultiHeadAttention_with_extra_mask(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None, attn_mask1=None, attn_mask2=None, attn_mask3=None):
        enc_output, enc_slf_attn, enc_hidden_states = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask, attn_mask1=attn_mask1, attn_mask2=attn_mask2, attn_mask3=attn_mask3)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn, enc_hidden_states


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()

        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(
            self, dec_input, enc_output,
            slf_attn_mask=None, dec_enc_attn_mask=None, cover=None):
        dec_output, dec_slf_attn, dec_hidden_states = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask, cover=None)
        dec_output, dec_enc_attn, enc_dec_contexts = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask, cover=cover)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn, dec_hidden_states, enc_dec_contexts
