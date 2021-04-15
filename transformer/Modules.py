import torch
import torch.nn as nn
import torch.nn.functional as F

__author__ = "Yu-Hsiang Huang"

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()

        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None, cover=None, score_matrix=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))


        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        if score_matrix is not None:
            attn *= score_matrix[:, :, :attn.shape[2], :]            


        attn = self.dropout(F.softmax(attn, dim = -1))



        if cover is not None:
            tensor_mask = torch.ones(attn.size(0),attn.size(1),attn.size(-1),attn.size(-1)).to(device)
            tensor_mask1 = torch.tril(tensor_mask, diagonal=-2)
            tensor_mask2 = torch.tril(tensor_mask, diagonal=1)
            tensor_mask = tensor_mask2-tensor_mask1

            #for row in range(tensor_mask.size(-2)):
            #    for col in range(tensor_mask.size(-1)):
            #        if(row >= col):
            #            tensor_mask[:,:,row,col] = 0 
            #tensor_mask = tensor_mask.transpose(-2,-1) 

            cv = torch.matmul(attn,tensor_mask)


            min_term = torch.min(cv,attn)

            mean_term1 = torch.sum(min_term,2)
            current_cover = torch.sum(mean_term1,1)

            cover += current_cover
            #cover += torch.sum(torch.min(cv,attn),[0,1,2,3]) 

        output = torch.matmul(attn, v)

        return output, attn
