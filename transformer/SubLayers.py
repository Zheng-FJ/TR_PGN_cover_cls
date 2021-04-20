''' Define the sublayers in encoder/decoder layer '''
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformer.Modules import ScaledDotProductAttention

__author__ = "Yu-Hsiang Huang"

class MultiHeadAttention_with_extra_mask(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention1 = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.attention2 = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.attention3 = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.attention4 = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.linear1 = nn.Linear(d_k, d_k)
        self.linear2 = nn.Linear(d_k, d_k)
        self.linear3 = nn.Linear(d_k, d_k)
        self.linear4 = nn.Linear(d_k, d_k)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)


        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm1 = nn.LayerNorm(d_k, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(d_k, eps=1e-6)
        self.layer_norm3 = nn.LayerNorm(d_k, eps=1e-6)
        self.layer_norm4 = nn.LayerNorm(d_k, eps=1e-6)


    def forward(self, q, k, v, mask=None, attn_mask1=None, attn_mask2=None, attn_mask3=None, cover=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.
        if attn_mask1 is not None:
            attn_mask1 = attn_mask1.unsqueeze(1)
        if attn_mask2 is not None:
            attn_mask2 = attn_mask2.unsqueeze(1)
        if attn_mask3 is not None:
            attn_mask3 = attn_mask3.unsqueeze(1)

        q1, attn1 = self.attention1(q, k, v, mask=mask)
        q2, attn2 = self.attention2(q, k, v, mask=attn_mask1)
        q3, attn3 = self.attention3(q, k, v, mask=attn_mask2) 
        q4, attn4 = self.attention4(q, k, v, mask=attn_mask3)   

        q1 = self.layer_norm1(self.dropout1(self.linear1(q1)))
        q2 = self.layer_norm2(self.dropout2(self.linear2(q2)))
        q3 = self.layer_norm3(self.dropout3(self.linear3(q3))) 
        q4 = self.layer_norm4(self.dropout4(self.linear4(q4)))  

        q = 0.5*q1 + 0.5*q3
        # q = 0.2*q1 + 0.4*q2 + 0.4*q3
        # q = q1
        attn = 0.5*attn1 + 0.5*attn3
        # attn = 0.2*attn1 + 0.4*attn2 + 0.4*attn3
        # attn = attn1


        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        output_plus_residual = q + residual

        output_plus_residual = self.layer_norm(output_plus_residual)

        return output_plus_residual, attn, q


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)




    def forward(self, q, k, v, score_matrix, mask=None, cover=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        output, attn= self.attention(q, k, v, mask=mask, cover=cover, score_matrix = score_matrix)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output_plus_residual = output + residual

        output_plus_residual = self.layer_norm(output_plus_residual)

        return output_plus_residual, attn, output





class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x
