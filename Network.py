from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F
from position_embedding import SinusoidalPositionalEmbedding
import math
from torch.autograd import Variable
import numpy as np

dim_in=283
dim_k=80
dim_v=80
dim_in_x=768
dim_in_y=283

class MultiHeadSelfAttention(nn.Module):
    dim_in: int  # input dimension
    dim_k: int   # key and query dimension
    dim_v: int   # value dimension
    num_heads: int  # number of heads, for each head, dim_* = dim_* // num_heads

    def __init__(self, length,dim_in, dim_k, dim_v, num_heads=6):
        super(MultiHeadSelfAttention, self).__init__()
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k // num_heads)
        self.layer_norm=nn.LayerNorm(dim_k)
        self.dropout=nn.Dropout(p=0.1)
        self.fc_dim1=dim_v*length #sequnece length
        self.fc_dim2=192
        self.fc_dim3=32
        self.fc_dim4 = 8
        self.fc_dim5 = 2
        self.fc1 = nn.Linear(self.fc_dim1, self.fc_dim2)
        self.fc2 = nn.Linear(self.fc_dim2, self.fc_dim3)
        self.fc3 = nn.Linear(self.fc_dim3, self.fc_dim4)
        self.fc4 = nn.Linear(self.fc_dim4, self.fc_dim5)
        self.bn1 = nn.BatchNorm1d(num_features=self.fc_dim2, eps=1e-05, momentum=0.1, affine=True,
                                 track_running_stats=True)
        self.bn2 = nn.BatchNorm1d(num_features=self.fc_dim3, eps=1e-05, momentum=0.1, affine=True,
                                 track_running_stats=True)
        self.bn3 = nn.BatchNorm1d(num_features=self.fc_dim4, eps=1e-05, momentum=0.1, affine=True,
                                 track_running_stats=True)

    def forward(self, x,attn_mask):
        # x: tensor of shape (batch, n, dim_in)
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        nh = self.num_heads
        dk = self.dim_k // nh  # dim_k of each head
        dv = self.dim_v // nh  # dim_v of each head

        q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        v = self.linear_v(x).reshape(batch, n, nh, dv).transpose(1, 2)  # (batch, nh, n, dv)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact  # batch, nh, n, n
        #dist=dist.masked_fill_(attn_mask, -1e9)
        dist = torch.softmax(dist, dim=-1)  # batch, nh, n, n
        dist=self.dropout(dist)
        att = torch.matmul(dist, v)  # batch, nh, n, dv
        att = att.transpose(1, 2).reshape(batch, n, self.dim_v)  # batch, n, dim_v
        #att_mask=att.eq(0)
        #att=att.masked_fill_(att_mask,-1e9)
        result=att
        """""
        att=self.layer_norm(att)
        att = torch.reshape(att, (att.shape[0], -1))
        att = self.fc1(att)
        att = F.gelu(att)
        att = self.bn1(att)
        att = F.dropout(att, p=0.4)
        att = self.fc2(att)
        att = F.gelu(att)
        att = self.bn2(att)
        att = F.dropout(att, p=0.4)
        att = self.fc3(att)
        att = F.gelu(att)
        att = self.bn3(att)
        att = F.dropout(att, p=0.4)
        att = self.fc4(att)
        """
        return result

    def get_attention_padding_mask(self, q):
        attn_pad_mask = q[:,:,0].eq(0).unsqueeze(1).repeat(1, q.size(1), 1)
        # |attn_pad_mask| : (batch_size, q_len, k_len)

        return attn_pad_mask




class MultiHeadCrossSelfAttention(nn.Module):
    dim_in: int  # input dimension
    dim_k: int  # key and query dimension
    dim_v: int  # value dimension
    num_heads: int  # number of heads, for each head, dim_* = dim_* // num_heads

    def __init__(self, length_x,length_y,dim_in_x,dim_in_y, dim_k, dim_v, num_heads=6):
        super(MultiHeadCrossSelfAttention, self).__init__()
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_in_x = dim_in_x
        self.dim_in_y=dim_in_y
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.length_y=length_y
        self.num_heads = num_heads
        self.linear_q = nn.Linear(dim_in_x, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in_y, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in_y, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k // num_heads)
        self.dropout=nn.Dropout(p=0.5)
        self.fc_dim1 = dim_v * length_x
        self.fc_dim2 = 192
        self.fc_dim3 = 32
        self.fc_dim4 = 8
        self.fc_dim5 = 2
        self.fc1 = nn.Linear(self.fc_dim1, self.fc_dim2)
        self.fc2 = nn.Linear(self.fc_dim2, self.fc_dim3)
        self.fc3 = nn.Linear(self.fc_dim3, self.fc_dim4)
        self.fc4 = nn.Linear(self.fc_dim4, self.fc_dim5)
        self.bn1 = nn.BatchNorm1d(num_features=self.fc_dim2, eps=1e-05, momentum=0.1, affine=True,
                                  track_running_stats=True)
        self.bn2 = nn.BatchNorm1d(num_features=self.fc_dim3, eps=1e-05, momentum=0.1, affine=True,
                                  track_running_stats=True)
        self.bn3 = nn.BatchNorm1d(num_features=self.fc_dim4, eps=1e-05, momentum=0.1, affine=True,
                                  track_running_stats=True)

    def forward(self, x,y,attn_mask):
        # x: tensor of shape (batch, n, dim_in)
        batch, n_x, dim_in_x = x.shape
        #batch, n_y, dim_in_y = y.shape
        assert dim_in_x == self.dim_in_x
        #print(y.shape)
        #print(dim_in_x)

        nh = self.num_heads
        dk = self.dim_k // nh  # dim_k of each head
        dv = self.dim_v // nh  # dim_v of each head

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        q = self.linear_q(x).reshape(batch, n_x, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        k = self.linear_k(y).reshape(batch, self.length_y, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        v = self.linear_v(y).reshape(batch, self.length_y, nh, dv).transpose(1, 2)  # (batch, nh, n, dv)

        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact  # batch, nh, n, n
        #dist = dist.masked_fill_(attn_mask, -1e9)
        #print(attn_mask[0,0,0,:])
        dist = torch.softmax(dist, dim=-1)  # batch, nh, n, n
        dist_out=dist
        dist=self.dropout(dist)
        att = torch.matmul(dist, v)  # batch, nh, n, dv
        att = att.transpose(1, 2).reshape(batch, n_x, self.dim_v)  # batch, n, dim_v
        result = att
        """""
        att = torch.reshape(att, (att.shape[0], -1))
        att = self.fc1(att)
        att = F.relu(att)
        att = self.bn1(att)
        att = F.dropout(att, p=0.4)
        att = self.fc2(att)
        att = F.relu(att)
        att = self.bn2(att)
        att = F.dropout(att, p=0.4)
        att = self.fc3(att)
        att = F.relu(att)
        att = self.bn3(att)
        att = F.dropout(att, p=0.4)
        att = self.fc4(att)
        """
        return result,dist_out



class SimpleAttention(nn.Module):

    def __init__(self, input_dim):
        super(SimpleAttention, self).__init__()
        self.input_dim = input_dim
        self.scalar = nn.Linear(self.input_dim, 1, bias=False)

    def forward(self, M,attn_mask, x=None):
        """
        M -> (batch,seq_len,  vector)
        x -> dummy argument for the compatibility with MatchingAttention
        """
        scale = self.scalar(M)  #  batch,seq_len, 1
        #scale=scale.masked_fill_(attn_mask, -1e9)
        alpha = F.softmax(scale, dim=1).permute(0, 2, 1)  # batch, 1, seq_len
        attn_pool = torch.bmm(alpha, M)[:, 0, :]  # batch, vector

        return attn_pool


class MultiHeadContrastiveSelfAttention(nn.Module):
    dim_in: int  # input dimension
    dim_k: int  # key and query dimension
    dim_v: int  # value dimension
    num_heads: int  # number of heads, for each head, dim_* = dim_* // num_heads

    def __init__(self, length_x,length_y,dim_in_x,dim_in_y, dim_k, dim_v, num_heads=6):
        super(MultiHeadContrastiveSelfAttention, self).__init__()
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_in_x = dim_in_x
        self.dim_in_y=dim_in_y
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.length_y=length_y
        self.num_heads = num_heads
        self.linear_q = nn.Linear(dim_in_x, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in_y, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in_y, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k // num_heads)
        self.dropout=nn.Dropout(p=0.1)
        self.fc_dim1 = dim_v * length_x
        self.fc_dim2 = 192
        self.fc_dim3 = 32
        self.fc_dim4 = 8
        self.fc_dim5 = 2
        self.fc1 = nn.Linear(self.fc_dim1, self.fc_dim2)
        self.fc2 = nn.Linear(self.fc_dim2, self.fc_dim3)
        self.fc3 = nn.Linear(self.fc_dim3, self.fc_dim4)
        self.fc4 = nn.Linear(self.fc_dim4, self.fc_dim5)
        self.bn1 = nn.BatchNorm1d(num_features=self.fc_dim2, eps=1e-05, momentum=0.1, affine=True,
                                  track_running_stats=True)
        self.bn2 = nn.BatchNorm1d(num_features=self.fc_dim3, eps=1e-05, momentum=0.1, affine=True,
                                  track_running_stats=True)
        self.bn3 = nn.BatchNorm1d(num_features=self.fc_dim4, eps=1e-05, momentum=0.1, affine=True,
                                  track_running_stats=True)

    def forward(self, x,y):
        # x: tensor of shape (batch, n, dim_in)
        batch, n_x, dim_in_x = x.shape
        #batch, n_y, dim_in_y = y.shape
        assert dim_in_x == self.dim_in_x
        #print(y.shape)
        #print(dim_in_x)

        nh = self.num_heads
        dk = self.dim_k // nh  # dim_k of each head
        dv = self.dim_v // nh  # dim_v of each head

        q = self.linear_q(x).reshape(batch, n_x, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        k = self.linear_k(y).reshape(batch, self.length_y, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        v = self.linear_v(y).reshape(batch, self.length_y, nh, dv).transpose(1, 2)  # (batch, nh, n, dv)

        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact  # batch, nh, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, nh, n, n
        ones=dist.new_ones(dist.size())
        dist=ones-dist
        dist = torch.softmax(dist, dim=-1)
        dist=self.dropout(dist)
        att = q+torch.matmul(dist, v)  # batch, nh, n, dv
        att = att.transpose(1, 2).reshape(batch, n_x, self.dim_v)  # batch, n, dim_v
        result = att
        att = torch.reshape(att, (att.shape[0], -1))
        '''
        att = self.fc1(att)
        att = F.relu(att)
        att = self.bn1(att)
        att = F.dropout(att, p=0.4)
        att = self.fc2(att)
        att = F.relu(att)
        att = self.bn2(att)
        att = F.dropout(att, p=0.4)
        att = self.fc3(att)
        att = F.relu(att)
        att = self.bn3(att)
        att = F.dropout(att, p=0.4)
        att = self.fc4(att)
        '''
        return result

#Modality x->y
class CrossAttentionNetwork(nn.Module):
    def __init__(self,length_x,length_y,dim_in_x,dim_in_y,dim_k,dim_v):
        super(CrossAttentionNetwork, self).__init__()
        self.dim_in_x=dim_in_x
        self.dim_in_y=dim_in_y
        self.dim_k=dim_k
        self.dim_v=dim_v
        self.MHSA=MultiHeadSelfAttention(length_y,dim_in_x,dim_k,dim_v)
        self.MHSA_x = MultiHeadSelfAttention(length_x, dim_in_x, dim_k, dim_v)
        self.MHCSA=MultiHeadCrossSelfAttention(length_x,length_y,dim_k,dim_in_y,dim_k,dim_v)
        self.contrastive=MultiHeadContrastiveSelfAttention(length_x,length_y,dim_k,dim_k,dim_k,dim_v)
        self.MHSAL_x = MultiHeadSelfAttentionLayers(length_x, dim_in_x, dim_k, dim_v)
        self.MHSAL_y=MultiHeadSelfAttentionLayers(length_y, dim_in_y, dim_k, dim_v)
        self.bn_x = nn.BatchNorm1d(num_features=self.dim_in_x, eps=1e-05, momentum=0.1, affine=True,
                                  track_running_stats=True)
        self.bn_y = nn.BatchNorm1d(num_features=self.dim_in_y, eps=1e-05, momentum=0.1, affine=True,
                                  track_running_stats=True)
        self.layer_norm = nn.LayerNorm(dim_k)
        self.layer_norm_x=nn.LayerNorm(dim_in_x)
        self.layer_norm_y = nn.LayerNorm(dim_in_y)
        self.dropout = nn.Dropout(0.5)
        self.conv_x = nn.Conv1d(dim_in_x, dim_k, kernel_size=1)
        self.conv_y=nn.Conv1d(dim_in_y,dim_k,kernel_size=1)
        self.d_ff=300
        self.w1=nn.Linear(dim_v,self.d_ff)
        self.w2=nn.Linear(self.d_ff,dim_v)
        self.fc_dim1 = dim_v * length_x
        self.fc_dim2 = 192
        self.fc_dim3 = 32
        self.fc_dim4 = 8
        self.fc_dim5 = 2
        self.fc1 = nn.Linear(self.fc_dim1, self.fc_dim2)
        self.fc2 = nn.Linear(self.fc_dim2, self.fc_dim3)
        self.fc3 = nn.Linear(self.fc_dim3, self.fc_dim4)
        self.fc4 = nn.Linear(self.fc_dim4, self.fc_dim5)
        self.bn1 = nn.BatchNorm1d(num_features=self.fc_dim2, eps=1e-05, momentum=0.1, affine=True,
                                  track_running_stats=True)
        self.bn2 = nn.BatchNorm1d(num_features=self.fc_dim3, eps=1e-05, momentum=0.1, affine=True,
                                  track_running_stats=True)
        self.bn3 = nn.BatchNorm1d(num_features=self.fc_dim4, eps=1e-05, momentum=0.1, affine=True,
                                  track_running_stats=True)

    def forward(self, x,y):
        #x=self.layer_norm_x(x)
        #y=self.layer_norm_y(y)
        #x=self.dropout(x)
        #y=self.dropout(y)
        #result=self.MultiHeadCrossSelfAttention
        #MHCSA=self.MultiHeadCrossSelfAttention
        #MHSAL_x=self.MHSAL_x
        #MHSAL_y=self.MHSAL_y
        #x=self.MHSAL_x(x)
        #y=self.MHSAL_y(y)
        #att=result(x,y)
        #att=MHCSA(x,y)
        #'''
        x=x.transpose(1,2)
        x=self.conv_x(x)
        x=x.transpose(1,2)
        residual=x
        #y=y.transpose(1,2)
        #y=self.conv_y(y)
        #y=y.transpose(1,2)
        x=self.layer_norm(x)
        #y=self.layer_norm(y)
        x=self.dropout(x)
        #y=self.dropout(y)
        residual=x
        #'''
        #x=self.MHSA(x)
        att=self.MHCSA(x,y)
        att=self.layer_norm(att+x)
        att=self.dropout(att)
        #contrastive=self.contrastive(x,y)
        #contrastive=self.layer_norm(contrastive)
        #contrastive=self.dropout(contrastive)
        #residual=att
        #att=self.MHSA(att)
        #att=self.layer_norm(att+residual)
        #att=self.dropout(att)
        #contrastive=self.MHSA(contrastive)
        #contrastive=self.layer_norm(contrastive)
        #contrastive=self.dropout(contrastive)
        #att=self.dropout(att)
        att = torch.reshape(att, (att.shape[0], -1))
        #contrastive=torch.reshape(contrastive,(contrastive.shape[0],-1))
        #att=torch.cat((att,contrastive),dim=1)
        att = self.fc1(att)
        #att = F.relu(att)
        att = self.bn1(att)
        att = F.dropout(att, p=0.5,training=self.training)
        att = self.fc2(att)
        #att = F.relu(att)
        att = self.bn2(att)
        att = F.dropout(att, p=0.5,training=self.training)
        att = self.fc3(att)
        #att = F.relu(att)
        att = self.bn3(att)
        att = F.dropout(att, p=0.5,training=self.training)
        att = self.fc4(att)

        return att

class Classify(nn.Module):
    dim_in: int  # input dimension
    dim_k: int   # key and query dimension
    dim_v: int   # value dimension
    num_heads: int  # number of heads, for each head, dim_* = dim_* // num_heads

    def __init__(self, length,dim_in):
        super(Classify, self).__init__()
        self.fc_dim1=dim_in*length #sequnece length
        self.fc_dim2=192
        self.fc_dim3=32
        self.fc_dim4 = 8
        self.fc_dim5 = 2
        self.fc1 = nn.Linear(self.fc_dim1, self.fc_dim2)
        self.fc2 = nn.Linear(self.fc_dim2, self.fc_dim3)
        self.fc3 = nn.Linear(self.fc_dim3, self.fc_dim4)
        self.fc4 = nn.Linear(self.fc_dim4, self.fc_dim5)
        self.bn1 = nn.BatchNorm1d(num_features=self.fc_dim2, eps=1e-05, momentum=0.1, affine=True,
                                 track_running_stats=True)
        self.bn2 = nn.BatchNorm1d(num_features=self.fc_dim3, eps=1e-05, momentum=0.1, affine=True,
                                 track_running_stats=True)
        self.bn3 = nn.BatchNorm1d(num_features=self.fc_dim4, eps=1e-05, momentum=0.1, affine=True,
                                 track_running_stats=True)

    def forward(self, att):
        # x: tensor of shape (batch, n, dim_in)
        att = torch.reshape(att, (att.shape[0], -1))
        att = self.fc1(att)
        att = F.relu(att)
        att = self.bn1(att)
        att = F.dropout(att, p=0.4)
        att = self.fc2(att)
        att = F.relu(att)
        att = self.bn2(att)
        att = F.dropout(att, p=0.4)
        att = self.fc3(att)
        att = F.relu(att)
        att = self.bn3(att)
        att = F.dropout(att, p=0.4)
        att = self.fc4(att)
        return att

class Classify_FC(nn.Module):
    dim_in: int  # input dimension
    dim_k: int   # key and query dimension
    dim_v: int   # value dimension
    num_heads: int  # number of heads, for each head, dim_* = dim_* // num_heads

    def __init__(self, dim1,dim2,dim3,dim4,dim5,dropout):
        super(Classify_FC, self).__init__()
        self.fc_dim1=dim1 #sequnece length
        self.fc_dim2=dim2
        self.fc_dim3=dim3
        self.fc_dim4 =dim4
        self.fc_dim5 =dim5
        self.fc1 = nn.Linear(self.fc_dim1, self.fc_dim2)
        self.fc2 = nn.Linear(self.fc_dim2, self.fc_dim3)
        self.fc3 = nn.Linear(self.fc_dim3, self.fc_dim4)
        self.fc4 = nn.Linear(self.fc_dim4, self.fc_dim5)
        self.bn1 = nn.BatchNorm1d(num_features=self.fc_dim2, eps=1e-05, momentum=0.1, affine=True,
                                 track_running_stats=True)
        self.bn2 = nn.BatchNorm1d(num_features=self.fc_dim3, eps=1e-05, momentum=0.1, affine=True,
                                 track_running_stats=True)
        self.bn3 = nn.BatchNorm1d(num_features=self.fc_dim4, eps=1e-05, momentum=0.1, affine=True,
                                 track_running_stats=True)
        self.drop1=nn.Dropout(p=0.2)
        self.drop2=nn.Dropout(p=0.2)
        self.dropout=dropout
    def forward(self, att):
        # x: tensor of shape (batch, n, dim_in)
        #att = torch.reshape(att, (att.shape[0], -1))
        #att=att.transpose(1,0)
        att = self.fc1(att)
        #att = F.relu(att)
        att = self.bn1(att)
        att = F.dropout(att, p=self.dropout,training=self.training)
        att = self.fc2(att)
        #att = F.relu(att)
        att = self.bn2(att)
        att = F.dropout(att, p=self.dropout,training=self.training)
        att = self.fc3(att)
        #att = F.relu(att)
        att = self.bn3(att)
        att = F.dropout(att, p=self.dropout,training=self.training)
        att = self.fc4(att)

        return att

class ContrastiveAttentionNetwork(nn.Module):
    def __init__(self,length_x,length_y,dim_in_x,dim_in_y,dim_k,dim_v):
        super(ContrastiveAttentionNetwork, self).__init__()
        self.dim_in_x=dim_in_x
        self.dim_in_y=dim_in_y
        self.dim_k=dim_k
        self.dim_v=dim_v
        self.MultiHeadSelfAttention=MultiHeadSelfAttention(length_y,dim_in_y,dim_k,dim_v)
        self.MultiHeadContrastiveSelfAttention=MultiHeadContrastiveSelfAttention(length_x,length_y,dim_in_x,dim_in_y,dim_k,dim_v)
        self.fc_dim1 = dim_v * length_x
        self.fc_dim2 = 192
        self.fc_dim3 = 32
        self.fc_dim4 = 8
        self.fc_dim5 = 2
        self.fc1 = nn.Linear(self.fc_dim1, self.fc_dim2)
        self.fc2 = nn.Linear(self.fc_dim2, self.fc_dim3)
        self.fc3 = nn.Linear(self.fc_dim3, self.fc_dim4)
        self.fc4 = nn.Linear(self.fc_dim4, self.fc_dim5)
        self.bn1 = nn.BatchNorm1d(num_features=self.fc_dim2, eps=1e-05, momentum=0.1, affine=True,
                                  track_running_stats=True)
        self.bn2 = nn.BatchNorm1d(num_features=self.fc_dim3, eps=1e-05, momentum=0.1, affine=True,
                                  track_running_stats=True)
        self.bn3 = nn.BatchNorm1d(num_features=self.fc_dim4, eps=1e-05, momentum=0.1, affine=True,
                                  track_running_stats=True)

    def forward(self, x,y):
        #y_att=self.MultiHeadSelfAttention
        #y_att=y_att(y)
        result=self.MultiHeadContrastiveSelfAttention
        att=result(x,y)
        att = self.fc1(att)
        att = F.relu(att)
        att = self.bn1(att)
        att = F.dropout(att, p=0.4)
        att = self.fc2(att)
        att = F.relu(att)
        att = self.bn2(att)
        att = F.dropout(att, p=0.4)
        att = self.fc3(att)
        att = F.relu(att)
        att = self.bn3(att)
        att = F.dropout(att, p=0.4)
        att = self.fc4(att)

        return att

class MultiHeadSelfAttentionLayers(nn.Module):
    dim_in: int  # input dimension
    dim_k: int   # key and query dimension
    dim_v: int   # value dimension
    num_heads: int  # number of heads, for each head, dim_* = dim_* // num_heads

    def __init__(self, length,dim_in, dim_k, dim_v, num_heads=6):
        super(MultiHeadSelfAttentionLayers, self).__init__()
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.MHSA1 = MultiHeadSelfAttention(length, dim_v, dim_k, dim_v)
        self.MHSA2 = MultiHeadSelfAttention(length, dim_v, dim_k, dim_v)
        self.MHSA3 = MultiHeadSelfAttention(length, dim_v, dim_k, dim_v)
        self.dropout=nn.Dropout(p=0.4)
        self.dropout1=nn.Dropout(p=0.4)
        self.dropout2 = nn.Dropout(p=0.4)
        self.layer_norm=nn.LayerNorm(dim_k)
        self.layer_norm_input= nn.LayerNorm(dim_in)
        self.conv_A = nn.Conv1d(dim_in, dim_k, kernel_size=1)
        self.conv_1=nn.Conv1d(dim_in, 512, kernel_size=1)
        self.conv_2=nn.Conv1d(dim_in, 256, kernel_size=1)
        self.conv_3=nn.Conv1d(dim_in, 150, kernel_size=1)
        self.fc_dim1=dim_v #sequnece length
        self.d_ff=300
        self.w1=nn.Linear(dim_v,self.d_ff)
        self.w2=nn.Linear(self.d_ff,dim_v)
        self.gru = torch.nn.GRU(dim_v, dim_v, 1,batch_first=True,dropout=0.5)
        self.fc_dim2=192
        self.fc_dim3=32
        self.fc_dim4 = 8
        self.fc_dim5 = 2
        self.fc1 = nn.Linear(self.fc_dim1, self.fc_dim2)
        self.fc2 = nn.Linear(self.fc_dim2, self.fc_dim3)
        self.fc3 = nn.Linear(self.fc_dim3, self.fc_dim4)
        self.fc4 = nn.Linear(self.fc_dim4, self.fc_dim5)
        self.ffn_1=FFN(self.dim_v)
        self.ffn_2=FFN(self.dim_v)
        self.bn=nn.BatchNorm1d(self.dim_in)
        self.bn1 = nn.BatchNorm1d(num_features=self.fc_dim2, eps=1e-05, momentum=0.1, affine=True,
                                 track_running_stats=True)
        self.bn2 = nn.BatchNorm1d(num_features=self.fc_dim3, eps=1e-05, momentum=0.1, affine=True,
                                 track_running_stats=True)
        self.bn3 = nn.BatchNorm1d(num_features=self.fc_dim4, eps=1e-05, momentum=0.1, affine=True,
                                 track_running_stats=True)
        self.SA=SimpleAttention(self.dim_v)

    def forward(self, x):
        # x: tensor of shape (batch, n, dim_in)
        #x=self.MHSA1(x)
        att_mask=self.get_attention_padding_mask_self(x)
        att_mask_simple=self.get_attention_padding_mask_simple(x)
        x=self.layer_norm_input(x)
        x=self.dropout(x)
        #residual=x
        x=x.transpose(1,2)
        x=self.conv_A(x)
        #x=F.relu(x)
        x=x.transpose(1,2)
        #x=self.layer_norm(x)
        x=self.dropout(x)
        residual=x
        output=x
        #gru
        #x,_=self.gru(x)
        #residual=x
        #Att Layer
        x=self.MHSA2(x,att_mask)
        x=self.layer_norm(x+residual)
        x=self.dropout1(x)
        #x=self.ffn_1(x)
        residual = x
        #FFN
        #x=self.w2(self.dropout(F.relu(self.w1(x))))
        #x=self.layer_norm(x+residual)
        #x=self.dropout(x)
        #Att Layer
        x=self.MHSA3(x,att_mask)
        x=self.layer_norm(x+residual)
        x=self.dropout2(x)
        #x=self.ffn_2(x)
        result=x
        #FFN
        #x=self.w2(self.dropout(F.relu(self.w1(x))))
        #x=self.layer_norm(x+residual)
        #x=self.dropout(x)
        #FC
        #x=torch.mean(x,dim=1)
        #x = torch.reshape(x, (x.shape[0], -1))
        x=self.SA(x,att_mask_simple)
        x = self.fc1(x)
        #x = F.relu(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        #x = F.relu(x)
        x = self.bn2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        #x = F.relu(x)
        x = self.bn3(x)
        x = self.dropout(x)
        x = self.fc4(x)
        return x

    def get_attention_padding_mask_self(self, q):
        attn_pad_mask = q[:, :, 0].eq(0).unsqueeze(1).repeat(1, q.size(1), 1)
        attn_pad_mask=attn_pad_mask.transpose(2,1)
        # |attn_pad_mask| : (batch_size, q_len, q_len)

        return attn_pad_mask

    def get_attention_padding_mask_simple(self, q):
        attn_pad_mask = q[:, :, 0].eq(0).unsqueeze(1)
        attn_pad_mask=attn_pad_mask.transpose(2,1)
        # |attn_pad_mask| : (batch_size, q_len, 1)

        return attn_pad_mask

class FFN(nn.Module):
    def __init__(self,dim_v):
        super(FFN, self).__init__()
        self.d_ff=300
        self.w1=nn.Linear(dim_v,self.d_ff)
        self.w2=nn.Linear(self.d_ff,dim_v)
        self.layernorm=nn.LayerNorm(dim_v)
        self.dropout=nn.Dropout(p=0.5)


    def forward(self,x):
        residual=x
        x=self.w2(self.dropout(self.w1(x)))
        x=self.layernorm(x+residual)
        x=self.dropout(x)

        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        """
        :param d_model: pe编码维度，一般与word embedding相同，方便相加
        :param dropout: dorp out
        :param max_len: 语料库中最长句子的长度，即word embedding中的L
        """
        super(PositionalEncoding, self).__init__()
        # 定义drop out
        self.dropout = nn.Dropout(p=dropout)
        # 计算pe编码
        pe = torch.zeros(max_len, d_model) # 建立空表，每行代表一个词的位置，每列代表一个编码位
        position = torch.arange(0, max_len).unsqueeze(1) # 建个arrange表示词的位置以便公式计算，size=(max_len,1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *    # 计算公式中10000**（2i/d_model)
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 计算偶数维度的pe值
        pe[:, 1::2] = torch.cos(position * div_term)  # 计算奇数维度的pe值
        pe = pe.unsqueeze(0)  # size=(1, L, d_model)，为了后续与word_embedding相加,意为batch维度下的操作相同
        self.register_buffer('pe', pe)  # pe值是不参加训练的

    def forward(self, x):
        # 输入的最终编码 = word_embedding + positional_embedding
        x = x + Variable(self.pe[:, :x.size(1)],requires_grad=False) #size = [batch, L, d_model]
        return self.dropout(x) # size = [batch, L, d_model]

class BiCrossmodalTransformer(nn.Module):
    def __init__(self,length_x,length_y,dim_v):
        super(BiCrossmodalTransformer, self).__init__()
        self.n_x=length_x
        self.n_y=length_y
        self.dim_v=dim_v
        self.MHSCA = MultiHeadCrossSelfAttention(self.n_x, self.n_y,self.dim_v, self.dim_v, self.dim_v, self.dim_v)
        self.dropout=nn.Dropout(p=0.5)
        self.layernorm=nn.LayerNorm(dim_v)
        self.FFN=FFN(dim_v)

    def forward(self,x,y,att_mask):
        out=self.MHSCA(x,y,att_mask)
        out=self.layernorm(out+x)
        out=self.dropout(out)
        #out=self.FFN(out)

        return out


class TriCrossmodalTransformer(nn.Module):
    def __init__(self, length_x, length_y,length_z,dim_v):
        super(TriCrossmodalTransformer, self).__init__()
        self.n_x = length_x
        self.n_y = length_y
        self.n_z=length_z
        self.biCT_1=BiCrossmodalTransformer(self.n_x,self.n_y,dim_v)
        self.biCT_2=BiCrossmodalTransformer(self.n_x,self.n_z,dim_v)

    def forward(self, x, y,z, att_mask_1,att_mask_2):
        out1=self.biCT_1(x,y,att_mask_1)
        out2= self.biCT_1(x, y, att_mask_1)


        return out1,out2

class ChannelAttentionModul(nn.Module):  # 通道注意力模块
    def __init__(self, in_channel, r=0.5):  # channel为输入的维度, r为全连接层缩放比例->控制中间层个数
        super(ChannelAttentionModul, self).__init__()
        # 全局最大池化
        self.MaxPool = nn.AdaptiveMaxPool1d(1)

        self.fc_MaxPool = nn.Sequential(
            nn.Linear(in_channel, int(in_channel * r)),  # int(channel * r)取整数, 中间层神经元数至少为1, 如有必要可设为向上取整
            nn.ReLU(),
            nn.Linear(int(in_channel * r), in_channel),
            nn.Sigmoid(),
        )

        # 全局均值池化
        self.AvgPool = nn.AdaptiveAvgPool1d(1)

        self.fc_AvgPool = nn.Sequential(
            nn.Linear(in_channel, int(in_channel * r)),  # int(channel * r)取整数, 中间层神经元数至少为1, 如有必要可设为向上取整
            nn.ReLU(),
            nn.Linear(int(in_channel * r), in_channel),
            nn.Sigmoid(),
        )

        # 激活函数
        self.sigmoid = nn.Sigmoid()
        self.softmax=nn.Softmax(dim=-1)

    def forward(self, x):
        # 1.最大池化分支
        max_branch = self.MaxPool(x)
        # 送入MLP全连接神经网络, 得到权重
        max_in = max_branch.view(max_branch.size(0), -1)
        max_weight = self.fc_MaxPool(max_in)

        # 2.全局池化分支
        avg_branch = self.AvgPool(x)
        # 送入MLP全连接神经网络, 得到权重
        avg_in = avg_branch.view(avg_branch.size(0), -1)
        avg_weight = self.fc_AvgPool(avg_in)

        # MaxPool + AvgPool 激活后得到权重weight
        weight = max_weight + avg_weight
        weight = self.softmax(weight)

        # 将维度为b, c的weight, reshape成b, c, 1, 1 与 输入x 相乘
        h, w = weight.shape
        # 通道注意力Mc
        Mc = torch.reshape(weight, (h, w, 1, 1))

        # 乘积获得结果
        weight=weight.unsqueeze(1)
        x = torch.bmm(weight,x)


        return x

class Trimodal(nn.Module):
    def __init__(self,length_x,length_y,length_z,dim_in_x,dim_in_y,dim_in_z,dim_k,dim_v):
        super(Trimodal, self).__init__()
        self.n_T=length_x
        self.n_V=length_y
        self.n_A=length_z
        self.dim_T=dim_in_x
        self.dim_V=dim_in_y
        self.dim_A=dim_in_z
        self.dim_k=dim_k
        self.dim_v=dim_v
        #self.embed_positions_T = SinusoidalPositionalEmbedding(dim_v)
        #self.embed_positions_V = SinusoidalPositionalEmbedding(dim_v)
        #self.embed_positions_A = SinusoidalPositionalEmbedding(dim_v)
        self.embed_positions_T = PositionalEncoding(dim_v,0,96)
        self.embed_positions_V = PositionalEncoding(dim_v,0,471)
        self.embed_positions_A = PositionalEncoding(dim_v,0,1000)
        ###########Attention######
        self.MHSCA_TV=MultiHeadCrossSelfAttention(self.n_T,self.n_V,self.dim_k,self.dim_k,self.dim_k,self.dim_v)
        self.MHSCA_TA=MultiHeadCrossSelfAttention(self.n_T,self.n_A,self.dim_k,self.dim_k,self.dim_k,self.dim_v)
        #self.MHSCTA_TV = MultiHeadContrastiveSelfAttention(self.n_T, self.n_V, self.dim_k, self.dim_k, self.dim_k, self.dim_v)
        #self.MHSCTA_TA = MultiHeadContrastiveSelfAttention(self.n_T, self.n_A, self.dim_k, self.dim_k, self.dim_k,
                                                           #self.dim_v)
        self.MHSCA_VA = MultiHeadCrossSelfAttention(self.n_V, self.n_A, self.dim_k, self.dim_k, self.dim_k, self.dim_v)
        self.MHSCA_AT = MultiHeadCrossSelfAttention(self.n_A, self.n_T, self.dim_k, self.dim_k, self.dim_k, self.dim_v)
        self.MHSCA_AV = MultiHeadCrossSelfAttention(self.n_A, self.n_V, self.dim_k, self.dim_k, self.dim_k, self.dim_v)
        self.MHSCA_VT = MultiHeadCrossSelfAttention(self.n_V, self.n_T, self.dim_k, self.dim_k, self.dim_k, self.dim_v)

        self.MHSCA_TV_2=MultiHeadCrossSelfAttention(self.n_T,self.n_V,self.dim_k,self.dim_k,self.dim_k,self.dim_v)
        self.MHSCA_TA_2=MultiHeadCrossSelfAttention(self.n_T,self.n_A,self.dim_k,self.dim_k,self.dim_k,self.dim_v)
        self.MHSCA_VA_2 = MultiHeadCrossSelfAttention(self.n_V, self.n_A, self.dim_k, self.dim_k, self.dim_k, self.dim_v)
        self.MHSCA_AT_2 = MultiHeadCrossSelfAttention(self.n_A, self.n_T, self.dim_k, self.dim_k, self.dim_k, self.dim_v)
        self.MHSCA_AV_2 = MultiHeadCrossSelfAttention(self.n_A, self.n_V, self.dim_k, self.dim_k, self.dim_k, self.dim_v)
        self.MHSCA_VT_2 = MultiHeadCrossSelfAttention(self.n_V, self.n_T, self.dim_k, self.dim_k, self.dim_k, self.dim_v)
        #A,T,V
        self.MHSCA_A_MFCC=TriCrossmodalTransformer(18,self.n_T,self.n_V,self.dim_v)
        self.MHSCA_TA_MFCC1=BiCrossmodalTransformer(self.n_T,self.n_A,self.dim_v)

        self.MHSA_T=MultiHeadSelfAttention(self.n_T,self.dim_k,self.dim_k,self.dim_k)
        self.MHSA_V = MultiHeadSelfAttention(self.n_V, self.dim_k, self.dim_k, self.dim_k)
        self.MHSA_A = MultiHeadSelfAttention(self.n_A, self.dim_k, self.dim_k, self.dim_k)
        self.MHSA_A_MFCC = MultiHeadSelfAttention(self.n_A, self.dim_k, self.dim_k, self.dim_k)


         #######Simple Attention
        self.SA_T=SimpleAttention(self.dim_v*3)
        self.SA_V=SimpleAttention(self.dim_v*3)
        self.SA_A=SimpleAttention(self.dim_v*3)
        self.SA_A_MFCC= SimpleAttention(self.dim_v*3)

        self.SA_T_ori=SimpleAttention(self.dim_k)
        self.SA_V_ori=SimpleAttention(self.dim_k)
        self.SA_A_ori=SimpleAttention(self.dim_k)

        self.SA_T_context_ori = SimpleAttention(self.dim_k)

        self.CA_T=ChannelAttentionModul(self.n_T)
        self.CA_V=ChannelAttentionModul(self.n_V)
        self.CA_A=ChannelAttentionModul(self.n_A)

        ##########same size encoder
        self.conv_T=nn.Conv1d(dim_in_x,dim_v,kernel_size=1)
        self.conv_V = nn.Conv1d(dim_in_y, dim_v, kernel_size=1)
        self.conv_A = nn.Conv1d(dim_in_z, dim_v, kernel_size=1)
        self.conv_A_MFCC = nn.Conv1d(283, dim_v, kernel_size=1)
        self.conv_T_context = nn.Conv1d(dim_in_x, dim_v, kernel_size=1)
        self.dropout=nn.Dropout(p=0.5)
        self.layernorm=nn.LayerNorm(dim_v)
        self.layernorm_T = nn.LayerNorm(self.dim_T)
        self.layernorm_V = nn.LayerNorm(self.dim_V)
        self.layernorm_A = nn.LayerNorm(self.dim_A)
        self.layernorm_A_MFCC=nn.LayerNorm(283)
        self.layernorm_T_context=nn.LayerNorm(self.dim_T)

        ###FFN###
        self.FFN_TA=FFN(dim_v)
        self.FFN_TV=FFN(dim_v)
        self.FFN_VA=FFN(dim_v)
        self.FFN_VT= FFN(dim_v)
        self.FFN_AV = FFN(dim_v)
        self.FFN_AT = FFN(dim_v)

        ########shared encoder
        self.shared=nn.Linear(dim_v,dim_v)

        self.linear_T=nn.Linear(self.dim_T,dim_v)


        self.atmf_dense_1 = nn.Linear(450, 128)
        self.atmf_dense_2 = nn.Linear(128, 32)
        self.atmf_dense_3 = nn.Linear(32, 4)
        self.atmf_dense_4 = nn.Linear(4, 1)
        self.W_F = nn.Parameter(torch.rand(32, 450, 450))
        self.W_f = nn.Parameter(torch.rand(32, 450, 1))

        ###########classify

        self.GRU_T = torch.nn.GRU(dim_v, dim_v, 1, batch_first=True, dropout=0)
        self.GRU_V = torch.nn.GRU(dim_v, dim_v, 1, batch_first=True, dropout=0)
        self.GRU_A = torch.nn.GRU(dim_v, dim_v, 1, batch_first=True, dropout=0)
        #self.fc_dim1 = dim_v *6+self.dim_T+self.dim_V+self.dim_A
        self.fc_dim1 = dim_v * 9
        self.fc_dim2 = 192
        self.fc_dim3 = 32
        self.fc_dim4 = 8
        self.fc_dim5 = 2
        self.fc1 = nn.Linear(self.fc_dim1, self.fc_dim2)
        self.fc2 = nn.Linear(self.fc_dim2, self.fc_dim3)
        self.fc3 = nn.Linear(self.fc_dim3, self.fc_dim4)
        self.fc4 = nn.Linear(self.fc_dim4, self.fc_dim5)
        self.fcSI1= nn.Linear(self.fc_dim1, self.fc_dim2)
        self.fcSE1 = nn.Linear(self.fc_dim1, self.fc_dim2)
        self.fcSI2 = nn.Linear(self.fc_dim2, self.fc_dim3)
        self.fcSE2 = nn.Linear(self.fc_dim2, 9)
        self.fcSI3 = nn.Linear(self.fc_dim3, self.fc_dim4)
        self.fcSI4 = nn.Linear(self.fc_dim4, 9)
        self.classify_SI=Classify_FC(self.fc_dim1,self.fc_dim2,self.fc_dim3,self.fc_dim4,3,0.5)
        self.classify_SE=Classify_FC(self.fc_dim1,self.fc_dim2,self.fc_dim3,self.fc_dim4,3,0.5)
        self.classify_EI=Classify_FC(self.fc_dim1,self.fc_dim2,self.fc_dim3,16,9,0.5)
        self.classify_EE = Classify_FC(self.fc_dim1, self.fc_dim2, self.fc_dim3, 16, 9, 0.5)
        self.bn1 = nn.BatchNorm1d(num_features=self.fc_dim2, eps=1e-05, momentum=0.1, affine=True,
                                  track_running_stats=True)
        self.bn2 = nn.BatchNorm1d(num_features=self.fc_dim3, eps=1e-05, momentum=0.1, affine=True,
                                  track_running_stats=True)
        self.bn3 = nn.BatchNorm1d(num_features=self.fc_dim4, eps=1e-05, momentum=0.1, affine=True,
                                  track_running_stats=True)

    def forward(self, T, V,A,A_MFCC):


        #########get attention mask
        attn_mask_TV=self.get_attention_padding_mask(T,V)
        attn_mask_TA=self.get_attention_padding_mask(T,A)
        attn_mask_VA=self.get_attention_padding_mask(V,A)
        attn_mask_VT = self.get_attention_padding_mask(V, T)
        attn_mask_AT = self.get_attention_padding_mask(A, T)
        attn_mask_AV = self.get_attention_padding_mask(A, V)
        attn_mask_AT_MFCC = self.get_attention_padding_mask(A_MFCC,T)
        attn_mask_AV_MFCC = self.get_attention_padding_mask(A_MFCC,V)
        attn_mask_TA_MFCC = self.get_attention_padding_mask(T, A_MFCC)
        attn_mask_T_simple=self.get_attention_padding_mask_simple(T)
        attn_mask_V_simple = self.get_attention_padding_mask_simple(V)
        attn_mask_A_simple = self.get_attention_padding_mask_simple(A)
        attn_mask_A_MFCC_simple = self.get_attention_padding_mask_simple(A_MFCC)
        attn_mask_T_self=self.get_attention_padding_mask_self(T)
        attn_mask_V_self=self.get_attention_padding_mask_self(V)
        attn_mask_A_self=self.get_attention_padding_mask_self(A)
        attn_mask_A_MFCC_self = self.get_attention_padding_mask_self(A_MFCC)

        T=self.layernorm_T(T)
        V = self.layernorm_V(V)
        A = self.layernorm_A(A)
        A_MFCC = self.layernorm_A_MFCC(A_MFCC)


        T_residual=T
        V_residual=V
        A_residual=A

        ############projection
        T=T.transpose(1,2)
        V=V.transpose(1, 2)
        A=A.transpose(1, 2)
        A_MFCC = A_MFCC.transpose(1, 2)

        T=self.conv_T(T)
        V=self.conv_V(V)
        A=self.conv_A(A)
        A_MFCC=self.conv_A_MFCC(A_MFCC)


        T=T.transpose(1,2)
        V=V.transpose(1, 2)
        A=A.transpose(1, 2)
        A_MFCC=A_MFCC.transpose(1,2)
        '''
        self.shared_V=torch.mean(V,dim=1)
        self.shared_A=torch.mean(A,dim=1)
        self.shared_T=torch.mean(T,dim=1)
        self.diff_T=torch.mean(T,dim=1)
        self.diff_V=torch.mean(V,dim=1)
        self.diff_A=torch.mean(A,dim=1)
        '''
        T=self.layernorm(T)
        T=self.dropout(T)
        V=self.layernorm(V)
        V=self.dropout(V)
        A=self.layernorm(A)
        A=self.dropout(A)
        A_MFCC=self.layernorm(A_MFCC)
        A_MFCC=self.dropout(A_MFCC)

        #T=T+self.embed_positions_T(T[:,:,0])
        #V=V+self.embed_positions_V(V[:,:,0])
        #A=A+self.embed_positions_A(A[:,:,0])
        '''
        T,_=self.GRU_T(T)
        V,_=self.GRU_V(V)
        A,_=self.GRU_A(A)
        T=self.layernorm(T)
        V=self.layernorm(V)
        A=self.layernorm(A)
        '''

        T=self.MHSA_T(T,attn_mask_T_self)
        V=self.MHSA_T(V,attn_mask_V_self)
        A=self.MHSA_T(A,attn_mask_A_self)
        A_MFCC=self.MHSA_A_MFCC(A_MFCC,attn_mask_A_MFCC_self)
        #T=self.layernorm(T)
        #V=self.layernorm(V)
        #A=self.layernorm(A)

        residual_T=T
        '''
        T=self.embed_positions_T(T)
        V=self.embed_positions_V(V)
        A=self.embed_positions_A(A)
        T=self.layernorm(T)
        V=self.layernorm(V)
        A=self.layernorm(A)
        '''
        ##########6 crossmodal attention networks#########
        TA,dist_TA=self.MHSCA_TA(T,A,attn_mask_TA)
        #TA=self.MHSCTA_TA(T,A)
        TA=self.layernorm(TA+residual_T)
        TA=self.dropout(TA)
        TA=self.FFN_TA(TA)

        TV,dist_TV=self.MHSCA_TV(T,V,attn_mask_TV)
        #TV=self.MHSCTA_TV(T,V)
        TV=self.layernorm(TV+residual_T)
        TV=self.dropout(TV)
        TV=self.FFN_TV(TV)

        VA,_=self.MHSCA_VA(V,A,attn_mask_VA)
        VA=self.layernorm(VA+V)
        VA=self.dropout(VA)
        VA=self.FFN_VA(VA)

        VT,_=self.MHSCA_VT(V,T,attn_mask_VT)
        VT=self.layernorm(VT+V)
        VT=self.dropout(VT)
        VT=self.FFN_VT(VT)

        AT,_=self.MHSCA_AT(A,T,attn_mask_AT)
        AT=self.layernorm(AT+A)
        AT=self.dropout(AT)
        AT=self.FFN_AT(AT)

        AV,_=self.MHSCA_AV(A,V,attn_mask_AV)
        AV=self.layernorm(AV+A)
        AV=self.dropout(AV)
        AV=self.FFN_AV(AV)

        #AT_MFCC,AV_MFCC=self.MHSCA_A_MFCC(A_MFCC,T,V,attn_mask_AT_MFCC,attn_mask_AV_MFCC)
        #TA_MFCC=self.MHSCA_TA_MFCC1(T,A_MFCC,attn_mask_TA_MFCC)





        #########prefusion
        #T=self.SA_T_ori(T,attn_mask_T_simple)
        #V = self.SA_V_ori(V, attn_mask_V_simple)
        #A = self.SA_A_ori(A, attn_mask_A_simple)

        #T=self.MHSA_T(T,attn_mask_T_self)
        #V=self.MHSA_V(V,attn_mask_V_self)
        #A=self.MHSA_A(A,attn_mask_A_self)

        #T = torch.reshape(T, (T.shape[0], -1))
        #V = torch.reshape(V, (V.shape[0], -1))
        #A = torch.reshape(A, (A.shape[0], -1))
        #TV = torch.reshape(TV, (TV.shape[0], -1))
        #TA = torch.reshape(TA, (TA.shape[0], -1))
        #VT = torch.reshape(VT, (VT.shape[0], -1))
        #VA = torch.reshape(VA, (VA.shape[0], -1))
        #AT = torch.reshape(AT, (AT.shape[0], -1))
        #AV = torch.reshape(AV, (AV.shape[0], -1))
        #VA = torch.reshape(VA, (VA.shape[0], -1))

        '''
        fusion方式：
        1：conv1d特征在特征维度与另外两个对应crossmodal输出特征拼接
        2：conv1d特征attention加权为长度1后在全连接层前与另外两个对应crossmodal加权为长度1的输出特征拼接
        3：使用原始特征
        1相对有效
        对输入数据先进行layer normalization有效果
        20epoch左右test acc到达峰值 71.83
        加入FFN训练acc波动变小
        GRU dropout=0.1或0，0最好
        FFN 的dff为300
        加入位置编码无明显提升
        att_mask之前layernorm
        att_mask以q为基准
        注意力权重dropout=0.5
        '''
        final_T=torch.cat((TA,TV,T),dim=2)
        final_T=self.SA_T(final_T,attn_mask_T_simple)
        #final_T=self.CA_T(final_T)
        #final_T=torch.cat((final_T,T_context),dim=1)


        final_V=torch.cat((VT,VA,V),dim=2)
        final_V=self.SA_V(final_V,attn_mask_V_simple)
        #final_V=self.CA_V(final_V)
        #final_V=torch.cat((final_V,V),dim=1)

        final_A=torch.cat((AT,AV,A),dim=2)
        final_A=self.SA_A(final_A,attn_mask_A_simple)
        #final_A=self.CA_A(final_A)

        #final_A_MFCC=torch.cat((AT_MFCC,AV_MFCC,A_MFCC),dim=2)
        #final_A_MFCC=self.SA_A_MFCC(final_A_MFCC,attn_mask_A_MFCC_simple)
        #final_A=torch.cat((final_A,A),dim=1)
        #'''
        #######atmf#######
        #tensor of shape:(batch,3,3*dim_v)
        #tensor of shape:(batch,1,dim_v)
        final_T=torch.unsqueeze(final_T,1)
        final_V = torch.unsqueeze(final_V, 1)
        final_A = torch.unsqueeze(final_A, 1)
        #final_A_MFCC=torch.unsqueeze(final_A_MFCC,1)

        Trimodal=torch.cat((final_T,final_V,final_A),dim=1)
        Trimodal=self.atmf(Trimodal)
        Trimodal=torch.reshape(Trimodal,(Trimodal.shape[0],-1))
        #'''
        #Trimodal=torch.cat((final_T,final_V,final_A),dim=1)

        '''
        #final=T+TV+TA
        SI=self.fcSI1(Trimodal)
        SE=self.fcSE1(Trimodal)

        SI=self.bn1(SI)
        SE=self.bn1(SE)

        SI = F.dropout(SI, p=0.5, training=self.training)
        SE = F.dropout(SE, p=0.5, training=self.training)

        SI=self.fcSI2(SI)
        SE=self.fcSE2(SE)

        SI = self.bn2(SI)
        SI = F.dropout(SI, p=0.5, training=self.training)

        SI=self.fcSI3(SI)
        SI = self.bn3(SI)
        SI = F.dropout(SI, p=0.5, training=self.training)

        SI=self.fcSI4(SI)
        '''

        final = self.fc1(Trimodal)
        #final = F.relu(final)
        final = self.bn1(final)
        final = F.dropout(final, p=0.5,training=self.training)
        final = self.fc2(final)
        #final = F.relu(final)
        final = self.bn2(final)
        final = F.dropout(final, p=0.5,training=self.training)
        final = self.fc3(final)
        #final = F.relu(final)
        final = self.bn3(final)
        final = F.dropout(final, p=0.5,training=self.training)
        final = self.fc4(final)

        SI=self.classify_SI(Trimodal)
        SE=self.classify_SE(Trimodal)
        EI=self.classify_EI(Trimodal)
        EE=self.classify_EE(Trimodal)

        return final,Trimodal,SI,SE,EI,EE,dist_TA,dist_TV

    # input: tensor of Q,K
    # output: tensor of shape:(q_len,k_len)
    '''
    def get_attention_padding_mask(self, q, k):
        attn_pad_mask = q[:, :, 0].eq(0).unsqueeze(1).repeat(1, k.size(1), 1)
        attn_pad_mask=attn_pad_mask.transpose(2,1)
        # |attn_pad_mask| : (batch_size, q_len, k_len)

        return attn_pad_mask
    '''

    def get_attention_padding_mask(self, q, k):
        attn_pad_mask = k[:, :, 0].eq(0).unsqueeze(1).repeat(1, q.size(1), 1)
        #attn_pad_mask=attn_pad_mask.transpose(2,1)
        # |attn_pad_mask| : (batch_size, q_len, k_len)

        return attn_pad_mask

    def get_attention_padding_mask_simple(self, q):
        attn_pad_mask = q[:, :, 0].eq(0).unsqueeze(1)
        attn_pad_mask=attn_pad_mask.transpose(2,1)
        # |attn_pad_mask| : (batch_size, q_len, 1)

        return attn_pad_mask

    def get_attention_padding_mask_self(self, q):
        attn_pad_mask = q[:, :, 0].eq(0).unsqueeze(1).repeat(1, q.size(1), 1)
        attn_pad_mask=attn_pad_mask.transpose(2,1)
        # |attn_pad_mask| : (batch_size, q_len, q_len)

        return attn_pad_mask

    def atmf(self, mm_feature):

        s = self.atmf_dense_4(self.atmf_dense_3(self.atmf_dense_2(self.atmf_dense_1(mm_feature))))
        s = s.permute(0, 2, 1)

        s = F.softmax(s, dim=2) + 1

        mm_feature=mm_feature.permute(0,2,1)

        wei_fea = mm_feature * s
        output=wei_fea.permute(0,1,2)
        batch_size=mm_feature.shape[0]
        assert batch_size==self.W_F[:batch_size,:,:].shape[0]
        P_F = torch.tanh(torch.bmm(self.W_F[:batch_size,:,:], wei_fea))
        P_F = F.softmax(P_F, dim=2)

        gamma_f = torch.bmm(self.W_f[:batch_size,:,:].permute(0, 2, 1), P_F)
        gamma_f = gamma_f.permute(0, 2, 1)
        atmf_output = torch.bmm(wei_fea, gamma_f)
        atmf_output=atmf_output.permute(0,2,1)

        return output

class Trimodal_C(nn.Module):
    def __init__(self,length_x,length_y,length_z,dim_in_x,dim_in_y,dim_in_z,dim_k,dim_v):
        super(Trimodal_C, self).__init__()
        self.n_T=length_x
        self.n_V=length_y
        self.n_A=length_z
        self.dim_T=dim_in_x
        self.dim_V=dim_in_y
        self.dim_A=dim_in_z
        self.dim_k=dim_k
        self.dim_v=dim_v
        #self.embed_positions_T = SinusoidalPositionalEmbedding(dim_v)
        #self.embed_positions_V = SinusoidalPositionalEmbedding(dim_v)
        #self.embed_positions_A = SinusoidalPositionalEmbedding(dim_v)
        self.embed_positions_T = PositionalEncoding(dim_v,0,96)
        self.embed_positions_V = PositionalEncoding(dim_v,0,471)
        self.embed_positions_A = PositionalEncoding(dim_v,0,1000)
        ###########Attention######
        self.MHSCA_TV=MultiHeadCrossSelfAttention(self.n_T,self.n_V,self.dim_k,self.dim_k,self.dim_k,self.dim_v)
        self.MHSCA_TA=MultiHeadCrossSelfAttention(self.n_T,self.n_A,self.dim_k,self.dim_k,self.dim_k,self.dim_v)
        #self.MHSCTA_TV = MultiHeadContrastiveSelfAttention(self.n_T, self.n_V, self.dim_k, self.dim_k, self.dim_k, self.dim_v)
        #self.MHSCTA_TA = MultiHeadContrastiveSelfAttention(self.n_T, self.n_A, self.dim_k, self.dim_k, self.dim_k,
                                                           #self.dim_v)
        self.MHSCA_VA = MultiHeadCrossSelfAttention(self.n_V, self.n_A, self.dim_k, self.dim_k, self.dim_k, self.dim_v)
        self.MHSCA_AT = MultiHeadCrossSelfAttention(self.n_A, self.n_T, self.dim_k, self.dim_k, self.dim_k, self.dim_v)
        self.MHSCA_AV = MultiHeadCrossSelfAttention(self.n_A, self.n_V, self.dim_k, self.dim_k, self.dim_k, self.dim_v)
        self.MHSCA_VT = MultiHeadCrossSelfAttention(self.n_V, self.n_T, self.dim_k, self.dim_k, self.dim_k, self.dim_v)

        self.MHSCA_TV_2=MultiHeadCrossSelfAttention(self.n_T,self.n_V,self.dim_k,self.dim_k,self.dim_k,self.dim_v)
        self.MHSCA_TA_2=MultiHeadCrossSelfAttention(self.n_T,self.n_A,self.dim_k,self.dim_k,self.dim_k,self.dim_v)
        self.MHSCA_VA_2 = MultiHeadCrossSelfAttention(self.n_V, self.n_A, self.dim_k, self.dim_k, self.dim_k, self.dim_v)
        self.MHSCA_AT_2 = MultiHeadCrossSelfAttention(self.n_A, self.n_T, self.dim_k, self.dim_k, self.dim_k, self.dim_v)
        self.MHSCA_AV_2 = MultiHeadCrossSelfAttention(self.n_A, self.n_V, self.dim_k, self.dim_k, self.dim_k, self.dim_v)
        self.MHSCA_VT_2 = MultiHeadCrossSelfAttention(self.n_V, self.n_T, self.dim_k, self.dim_k, self.dim_k, self.dim_v)
        #A,T,V
        self.MHSCA_A_MFCC=TriCrossmodalTransformer(18,self.n_T,self.n_V,self.dim_v)

        self.MHSA_T=MultiHeadSelfAttention(self.n_T,self.dim_k,self.dim_k,self.dim_k)
        self.MHSA_V = MultiHeadSelfAttention(self.n_V, self.dim_k, self.dim_k, self.dim_k)
        self.MHSA_A = MultiHeadSelfAttention(self.n_A, self.dim_k, self.dim_k, self.dim_k)
        self.MHSA_A_MFCC = MultiHeadSelfAttention(self.n_A, self.dim_k, self.dim_k, self.dim_k)


         #######Simple Attention
        self.SA_T=SimpleAttention(self.dim_v*3)
        self.SA_V=SimpleAttention(self.dim_v*3)
        self.SA_A=SimpleAttention(self.dim_v*3)
        self.SA_A_MFCC= SimpleAttention(self.dim_v*3)

        self.SA_T_ori=SimpleAttention(self.dim_k)
        self.SA_V_ori=SimpleAttention(self.dim_k)
        self.SA_A_ori=SimpleAttention(self.dim_k)

        self.SA_T_context_ori = SimpleAttention(self.dim_k)

        ##########same size encoder
        self.conv_T=nn.Conv1d(dim_in_x,dim_v,kernel_size=1)
        self.conv_V = nn.Conv1d(dim_in_y, dim_v, kernel_size=1)
        self.conv_A = nn.Conv1d(dim_in_z, dim_v, kernel_size=1)
        self.conv_A_MFCC = nn.Conv1d(283, dim_v, kernel_size=1)
        self.conv_T_context = nn.Conv1d(dim_in_x, dim_v, kernel_size=1)
        self.dropout=nn.Dropout(p=0.5)
        self.layernorm=nn.LayerNorm(dim_v)
        self.layernorm_T = nn.LayerNorm(self.dim_T)
        self.layernorm_V = nn.LayerNorm(self.dim_V)
        self.layernorm_A = nn.LayerNorm(self.dim_A)
        self.layernorm_A_MFCC=nn.LayerNorm(283)
        self.layernorm_T_context=nn.LayerNorm(self.dim_T)

        ###FFN###
        self.FFN_TA=FFN(dim_v)
        self.FFN_TV=FFN(dim_v)
        self.FFN_VA=FFN(dim_v)
        self.FFN_VT= FFN(dim_v)
        self.FFN_AV = FFN(dim_v)
        self.FFN_AT = FFN(dim_v)

        ########shared encoder
        self.shared=nn.Linear(dim_v,dim_v)

        self.linear_T=nn.Linear(self.dim_T,dim_v)


        self.atmf_dense_1 = nn.Linear(450, 128)
        self.atmf_dense_2 = nn.Linear(128, 32)
        self.atmf_dense_3 = nn.Linear(32, 4)
        self.atmf_dense_4 = nn.Linear(4, 1)
        self.W_F = nn.Parameter(torch.rand(32, 450, 450))
        self.W_f = nn.Parameter(torch.rand(32, 450, 1))

        ###########classify
        self.GRU_T = torch.nn.GRU(dim_v, dim_v, 1, batch_first=True, dropout=0)
        self.GRU_V = torch.nn.GRU(dim_v, dim_v, 1, batch_first=True, dropout=0)
        self.GRU_A = torch.nn.GRU(dim_v, dim_v, 1, batch_first=True, dropout=0)
        #self.fc_dim1 = dim_v *6+self.dim_T+self.dim_V+self.dim_A
        self.fc_dim1 = dim_v * 9
        self.fc_dim2 = 192
        self.fc_dim3 = 32
        self.fc_dim4 = 8
        self.fc_dim5 = 2
        self.fc1 = nn.Linear(self.fc_dim1, self.fc_dim2)
        self.fc2 = nn.Linear(self.fc_dim2, self.fc_dim3)
        self.fc3 = nn.Linear(self.fc_dim3, self.fc_dim4)
        self.fc4 = nn.Linear(self.fc_dim4, self.fc_dim5)
        self.bn1 = nn.BatchNorm1d(num_features=self.fc_dim2, eps=1e-05, momentum=0.1, affine=True,
                                  track_running_stats=True)
        self.bn2 = nn.BatchNorm1d(num_features=self.fc_dim3, eps=1e-05, momentum=0.1, affine=True,
                                  track_running_stats=True)
        self.bn3 = nn.BatchNorm1d(num_features=self.fc_dim4, eps=1e-05, momentum=0.1, affine=True,
                                  track_running_stats=True)

    def forward(self, T, V,A):

        T=self.layernorm_T(T)
        V = self.layernorm_V(V)
        A = self.layernorm_A(A)
        #########get attention mask
        attn_mask_TV=self.get_attention_padding_mask(T,V)
        attn_mask_TA=self.get_attention_padding_mask(T,A)
        attn_mask_VA=self.get_attention_padding_mask(V,A)
        attn_mask_VT = self.get_attention_padding_mask(V, T)
        attn_mask_AT = self.get_attention_padding_mask(A, T)
        attn_mask_AV = self.get_attention_padding_mask(A, V)
        attn_mask_T_simple=self.get_attention_padding_mask_simple(T)
        attn_mask_V_simple = self.get_attention_padding_mask_simple(V)
        attn_mask_A_simple = self.get_attention_padding_mask_simple(A)
        attn_mask_T_self=self.get_attention_padding_mask_self(T)
        attn_mask_V_self=self.get_attention_padding_mask_self(V)
        attn_mask_A_self=self.get_attention_padding_mask_self(A)

        T_residual=T
        V_residual=V
        A_residual=A

        ############projection
        T=T.transpose(1,2)
        V=V.transpose(1, 2)
        A=A.transpose(1, 2)

        T=self.conv_T(T)
        V=self.conv_V(V)
        A=self.conv_A(A)


        T=T.transpose(1,2)
        V=V.transpose(1, 2)
        A=A.transpose(1, 2)
        '''
        self.shared_V=torch.mean(V,dim=1)
        self.shared_A=torch.mean(A,dim=1)
        self.shared_T=torch.mean(T,dim=1)
        self.diff_T=torch.mean(T,dim=1)
        self.diff_V=torch.mean(V,dim=1)
        self.diff_A=torch.mean(A,dim=1)
        '''
        T=self.layernorm(T)
        T=self.dropout(T)
        V=self.layernorm(V)
        V=self.dropout(V)
        A=self.layernorm(A)
        A=self.dropout(A)

        #T=T+self.embed_positions_T(T[:,:,0])
        #V=V+self.embed_positions_V(V[:,:,0])
        #A=A+self.embed_positions_A(A[:,:,0])
        '''
        T,_=self.GRU_T(T)
        V,_=self.GRU_V(V)
        A,_=self.GRU_A(A)
        T=self.layernorm(T)
        V=self.layernorm(V)
        A=self.layernorm(A)
        '''

        T=self.MHSA_T(T,attn_mask_T_self)
        V=self.MHSA_T(V,attn_mask_V_self)
        A=self.MHSA_T(A,attn_mask_A_self)
        #T=self.layernorm(T)
        #V=self.layernorm(V)
        #A=self.layernorm(A)

        residual_T=T
        '''
        T=self.embed_positions_T(T)
        V=self.embed_positions_V(V)
        A=self.embed_positions_A(A)
        T=self.layernorm(T)
        V=self.layernorm(V)
        A=self.layernorm(A)
        '''
        ##########6 crossmodal attention networks#########
        TA=self.MHSCA_TA(T,A,attn_mask_TA)
        #TA=self.MHSCTA_TA(T,A)
        TA=self.layernorm(TA+residual_T)
        TA=self.dropout(TA)
        TA=self.FFN_TA(TA)

        TV=self.MHSCA_TV(T,V,attn_mask_TV)
        #TV=self.MHSCTA_TV(T,V)
        TV=self.layernorm(TV+residual_T)
        TV=self.dropout(TV)
        TV=self.FFN_TV(TV)

        VA=self.MHSCA_VA(V,A,attn_mask_VA)
        VA=self.layernorm(VA+V)
        VA=self.dropout(VA)
        VA=self.FFN_VA(VA)

        VT=self.MHSCA_VT(V,T,attn_mask_VT)
        VT=self.layernorm(VT+V)
        VT=self.dropout(VT)
        VT=self.FFN_VT(VT)

        AT=self.MHSCA_AT(A,T,attn_mask_AT)
        AT=self.layernorm(AT+A)
        AT=self.dropout(AT)
        AT=self.FFN_AT(AT)

        AV=self.MHSCA_AV(A,V,attn_mask_AV)
        AV=self.layernorm(AV+A)
        AV=self.dropout(AV)
        AV=self.FFN_AV(AV)



        '''
        fusion方式：
        1：conv1d特征在特征维度与另外两个对应crossmodal输出特征拼接
        2：conv1d特征attention加权为长度1后在全连接层前与另外两个对应crossmodal加权为长度1的输出特征拼接
        3：使用原始特征
        1相对有效
        对输入数据先进行layer normalization有效果
        20epoch左右test acc到达峰值 71.83
        加入FFN训练acc波动变小
        GRU dropout=0.1或0，0最好
        FFN 的dff为300
        加入位置编码无明显提升
        '''
        final_T=torch.cat((TA,TV,T),dim=2)
        final_T=self.SA_T(final_T,attn_mask_T_simple)
        #final_T=torch.cat((final_T,T_context),dim=1)


        final_V=torch.cat((VT,VA,V),dim=2)
        final_V=self.SA_V(final_V,attn_mask_V_simple)
        #final_V=torch.cat((final_V,V),dim=1)

        final_A=torch.cat((AT,AV,A),dim=2)
        final_A=self.SA_A(final_A,attn_mask_A_simple)

        #final_A=torch.cat((final_A,A),dim=1)
        #'''
        #######atmf#######
        #tensor of shape:(batch,3,3*dim_v)
        #tensor of shape:(batch,1,dim_v)
        final_T=torch.unsqueeze(final_T,1)
        final_V = torch.unsqueeze(final_V, 1)
        final_A = torch.unsqueeze(final_A, 1)

        Trimodal=torch.cat((final_T,final_V,final_A),dim=1)
        Trimodal=self.atmf(Trimodal)
        Trimodal=torch.reshape(Trimodal,(Trimodal.shape[0],-1))
        #'''
        #Trimodal=torch.cat((final_T,final_V,final_A),dim=1)


        #final=T+TV+TA

        return Trimodal

    # input: tensor of Q,K
    # output: tensor of shape:(q_len,k_len)
    def get_attention_padding_mask(self, q, k):
        attn_pad_mask = q[:, :, 0].eq(0).unsqueeze(1).repeat(1, k.size(1), 1)
        attn_pad_mask=attn_pad_mask.transpose(2,1)
        # |attn_pad_mask| : (batch_size, q_len, k_len)

        return attn_pad_mask

    def get_attention_padding_mask_simple(self, q):
        attn_pad_mask = q[:, :, 0].eq(0).unsqueeze(1)
        attn_pad_mask=attn_pad_mask.transpose(2,1)
        # |attn_pad_mask| : (batch_size, q_len, 1)

        return attn_pad_mask

    def get_attention_padding_mask_self(self, q):
        attn_pad_mask = q[:, :, 0].eq(0).unsqueeze(1).repeat(1, q.size(1), 1)
        attn_pad_mask=attn_pad_mask.transpose(2,1)
        # |attn_pad_mask| : (batch_size, q_len, q_len)

        return attn_pad_mask

    def atmf(self, mm_feature):

        s = self.atmf_dense_4(self.atmf_dense_3(self.atmf_dense_2(self.atmf_dense_1(mm_feature))))
        s = s.permute(0, 2, 1)

        s = F.softmax(s, dim=2) + 1

        mm_feature=mm_feature.permute(0,2,1)

        wei_fea = mm_feature * s
        output=wei_fea.permute(0,1,2)
        batch_size=mm_feature.shape[0]
        assert batch_size==self.W_F[:batch_size,:,:].shape[0]
        P_F = torch.tanh(torch.bmm(self.W_F[:batch_size,:,:], wei_fea))
        P_F = F.softmax(P_F, dim=2)

        gamma_f = torch.bmm(self.W_f[:batch_size,:,:].permute(0, 2, 1), P_F)
        gamma_f = gamma_f.permute(0, 2, 1)
        atmf_output = torch.bmm(wei_fea, gamma_f)
        atmf_output=atmf_output.permute(0,2,1)

        return output

class Trimodal_Context(nn.Module):
    def __init__(self,length_x,length_y,length_z,dim_in_x,dim_in_y,dim_in_z,dim_k,dim_v):
        super(Trimodal_Context, self).__init__()
        self.n_T=length_x
        self.n_V=length_y
        self.n_A=length_z
        self.dim_T=dim_in_x
        self.dim_V=dim_in_y
        self.dim_A=dim_in_z
        self.dim_k=dim_k
        self.dim_v=dim_v
        self.GRU_dropout=0.1
        self.dim_hidden=128
        self.dim_fusion=256
        ###########Attention######
        self.MHSCA_TV=MultiHeadCrossSelfAttention(self.n_T,self.n_V,self.dim_k,self.dim_k,self.dim_k,self.dim_v)
        self.MHSCA_TA=MultiHeadCrossSelfAttention(self.n_T,self.n_A,self.dim_k,self.dim_k,self.dim_k,self.dim_v)
        self.MHSCTA_TV = MultiHeadContrastiveSelfAttention(self.n_T, self.n_V, self.dim_k, self.dim_k, self.dim_k, self.dim_v)
        self.MHSCTA_TA = MultiHeadContrastiveSelfAttention(self.n_T, self.n_A, self.dim_k, self.dim_k, self.dim_k,
                                                           self.dim_v)
        self.MHSCA_VA = MultiHeadCrossSelfAttention(self.n_V, self.n_A, self.dim_k, self.dim_k, self.dim_k, self.dim_v)
        self.MHSCA_AT = MultiHeadCrossSelfAttention(self.n_A, self.n_T, self.dim_k, self.dim_k, self.dim_k, self.dim_v)
        self.MHSCA_AV = MultiHeadCrossSelfAttention(self.n_A, self.n_V, self.dim_k, self.dim_k, self.dim_k, self.dim_v)
        self.MHSCA_VT = MultiHeadCrossSelfAttention(self.n_V, self.n_T, self.dim_k, self.dim_k, self.dim_k, self.dim_v)

        self.MHSA_T=MultiHeadSelfAttention(self.n_T,self.dim_k,self.dim_k,self.dim_k)
        self.MHSA_V = MultiHeadSelfAttention(self.n_V, self.dim_k, self.dim_k, self.dim_k)
        self.MHSA_A = MultiHeadSelfAttention(self.n_A, self.dim_k, self.dim_k, self.dim_k)
        self.MHSA_T_C=MultiHeadSelfAttention(self.n_T,self.dim_k,self.dim_k,self.dim_k)
        self.MHSA_V_C = MultiHeadSelfAttention(self.n_V, self.dim_k, self.dim_k, self.dim_k)
        self.MHSA_A_C = MultiHeadSelfAttention(self.n_A, self.dim_k, self.dim_k, self.dim_k)

        self.Context=Trimodal_C(self.n_T,self.n_V,self.n_A,self.dim_T,self.dim_V,self.dim_A,self.dim_k,self.dim_k)

         #######Simple Attention
        self.SA_T=SimpleAttention(self.dim_v*3)
        self.SA_V=SimpleAttention(self.dim_v*3)
        self.SA_A=SimpleAttention(self.dim_v*3)

        self.SA_T_ori=SimpleAttention(self.dim_k)
        self.SA_V_ori=SimpleAttention(self.dim_k)
        self.SA_A_ori=SimpleAttention(self.dim_k)
        self.SA_T_context_ori = SimpleAttention(self.dim_k)
        self.SA_V_context_ori = SimpleAttention(self.dim_k)
        self.SA_A_context_ori = SimpleAttention(self.dim_k)

        ##########same size encoder
        self.conv_T=nn.Conv1d(dim_in_x,dim_v,kernel_size=1)
        self.conv_V = nn.Conv1d(dim_in_y, dim_v, kernel_size=1)
        self.conv_A = nn.Conv1d(dim_in_z, dim_v, kernel_size=1)
        self.conv_T_context = nn.Conv1d(dim_in_x, dim_v, kernel_size=1)
        self.conv_V_context = nn.Conv1d(dim_in_y, dim_v, kernel_size=1)
        self.conv_A_context = nn.Conv1d(dim_in_z, dim_v, kernel_size=1)
        self.dropout=nn.Dropout(p=0.5)
        self.layernorm=nn.LayerNorm(dim_v)
        self.layernorm_T = nn.LayerNorm(self.dim_T)
        self.layernorm_V = nn.LayerNorm(self.dim_V)
        self.layernorm_A = nn.LayerNorm(self.dim_A)
        self.layernorm_T_context=nn.LayerNorm(self.dim_T)
        self.layernorm_V_context = nn.LayerNorm(self.dim_V)
        self.layernorm_A_context=nn.LayerNorm(self.dim_A)

        ###FFN###
        self.FFN_TA=FFN(dim_v)
        self.FFN_TV=FFN(dim_v)
        self.FFN_VA=FFN(dim_v)
        self.FFN_VT= FFN(dim_v)
        self.FFN_AV = FFN(dim_v)
        self.FFN_AT = FFN(dim_v)

        ########shared encoder
        self.shared=nn.Linear(dim_v,dim_v)

        self.linear_T=nn.Linear(self.dim_T,dim_v)

        self.subnet_T = nn.Sequential()
        self.subnet_T.add_module('layernorm', nn.LayerNorm(self.dim_T))
        self.subnet_T.add_module('dropout', nn.Dropout(0.2))
        self.subnet_T.add_module('1', nn.Linear(self.dim_T, self.dim_hidden))
        self.subnet_T.add_module('2', nn.Linear(self.dim_hidden, self.dim_hidden))
        self.subnet_T.add_module('3', nn.Linear(self.dim_hidden,self.dim_v))

        self.subnet_V = nn.Sequential()
        self.subnet_V.add_module('layernorm', nn.LayerNorm(self.dim_V))
        self.subnet_V.add_module('dropout', nn.Dropout(0.2))
        self.subnet_V.add_module('1', nn.Linear(self.dim_V, self.dim_hidden))
        self.subnet_V.add_module('2', nn.Linear(self.dim_hidden, self.dim_hidden))
        self.subnet_V.add_module('3', nn.Linear(self.dim_hidden,self.dim_v))

        self.subnet_A = nn.Sequential()
        self.subnet_A.add_module('layernorm', nn.LayerNorm(self.dim_A))
        self.subnet_A.add_module('dropout', nn.Dropout(0.2))
        self.subnet_A.add_module('1', nn.Linear(self.dim_A, self.dim_hidden))
        self.subnet_A.add_module('2', nn.Linear(self.dim_hidden, self.dim_hidden))
        self.subnet_A.add_module('3', nn.Linear(self.dim_hidden,self.dim_v))


        self.atmf_dense_1 = nn.Linear(450, 128)
        self.atmf_dense_2 = nn.Linear(128, 32)
        self.atmf_dense_3 = nn.Linear(32, 4)
        self.atmf_dense_4 = nn.Linear(4, 1)
        self.W_F = nn.Parameter(torch.rand(32, 450, 450))
        self.W_f = nn.Parameter(torch.rand(32, 450, 1))

        ###########classify
        self.GRU_T = torch.nn.GRU(dim_v, dim_v, 1, batch_first=True, dropout=self.GRU_dropout)
        self.GRU_V = torch.nn.GRU(dim_v, dim_v, 1, batch_first=True, dropout=self.GRU_dropout)
        self.GRU_A = torch.nn.GRU(dim_v, dim_v, 1, batch_first=True, dropout=self.GRU_dropout)
        self.GRU_T_context = torch.nn.GRU(dim_v, dim_v, 1, batch_first=True, dropout=self.GRU_dropout)
        self.GRU_V_context=torch.nn.GRU(dim_v,dim_v,1,batch_first=True,dropout=self.GRU_dropout)
        self.GRU_A_context=torch.nn.GRU(dim_v,dim_v,1,batch_first=True,dropout=self.GRU_dropout)
        #self.fc_dim1 = dim_v *6+self.dim_T+self.dim_V+self.dim_A
        self.fc_dim1 = dim_v*18
        self.fc_dim2 = 192
        self.fc_dim3 = 32
        self.fc_dim4 = 8
        self.fc_dim5 = 2
        self.fc1 = nn.Linear(self.fc_dim1, self.fc_dim2)
        self.fc2 = nn.Linear(self.fc_dim2, self.fc_dim3)
        self.fc3 = nn.Linear(self.fc_dim3, self.fc_dim4)
        self.fc4 = nn.Linear(self.fc_dim4, self.fc_dim5)
        self.bn1 = nn.BatchNorm1d(num_features=self.fc_dim2, eps=1e-05, momentum=0.1, affine=True,
                                  track_running_stats=True)
        self.bn2 = nn.BatchNorm1d(num_features=self.fc_dim3, eps=1e-05, momentum=0.1, affine=True,
                                  track_running_stats=True)
        self.bn3 = nn.BatchNorm1d(num_features=self.fc_dim4, eps=1e-05, momentum=0.1, affine=True,
                                  track_running_stats=True)

    def forward(self, T, V,A,T_C,V_C,A_C):

        T=self.layernorm_T(T)
        V = self.layernorm_V(V)
        A = self.layernorm_A(A)


        #########get attention mask
        attn_mask_TV=self.get_attention_padding_mask(T,V)
        attn_mask_TA=self.get_attention_padding_mask(T,A)
        attn_mask_VA=self.get_attention_padding_mask(V,A)
        attn_mask_VT = self.get_attention_padding_mask(V, T)
        attn_mask_AT = self.get_attention_padding_mask(A, T)
        attn_mask_AV = self.get_attention_padding_mask(A, V)
        attn_mask_T_simple=self.get_attention_padding_mask_simple(T)
        attn_mask_V_simple = self.get_attention_padding_mask_simple(V)
        attn_mask_A_simple = self.get_attention_padding_mask_simple(A)
        #attn_mask_T_context_simple = self.get_attention_padding_mask_simple(T_C)
        #attn_mask_V_context_simple = self.get_attention_padding_mask_simple(V_C)
        #attn_mask_A_context_simple = self.get_attention_padding_mask_simple(A_C)
        attn_mask_T_self=self.get_attention_padding_mask_self(T)
        attn_mask_V_self=self.get_attention_padding_mask_self(V)
        attn_mask_A_self=self.get_attention_padding_mask_self(A)
        attn_mask_T_C_self=self.get_attention_padding_mask_self(T_C)
        attn_mask_V_C_self=self.get_attention_padding_mask_self(V_C)
        attn_mask_A_C_self=self.get_attention_padding_mask_self(A_C)

        T_residual=T
        V_residual=V
        A_residual=A

        ############projection
        T=T.transpose(1,2)
        V=V.transpose(1, 2)
        A=A.transpose(1, 2)

        T=self.conv_T(T)
        V=self.conv_V(V)
        A=self.conv_A(A)

        T=T.transpose(1,2)
        V=V.transpose(1, 2)
        A=A.transpose(1, 2)


        T=self.layernorm(T)
        T=self.dropout(T)
        V=self.layernorm(V)
        V=self.dropout(V)
        A=self.layernorm(A)
        A=self.dropout(A)

        T=self.MHSA_T(T,attn_mask_T_self)
        V=self.MHSA_T(V,attn_mask_V_self)
        A=self.MHSA_T(A,attn_mask_A_self)
        #T,_=self.GRU_T(T)
        #V,_=self.GRU_V(V)
        #A,_=self.GRU_A(A)

        residual_T=T

        ##########6 crossmodal attention networks#########
        TA=self.MHSCA_TA(T,A,attn_mask_TA)
        #TA=self.MHSCTA_TA(T,A)
        TA=self.layernorm(TA+residual_T)
        TA=self.dropout(TA)
        TA=self.FFN_TA(TA)

        TV=self.MHSCA_TV(T,V,attn_mask_TV)
        #TV=self.MHSCTA_TV(T,V)
        TV=self.layernorm(TV+residual_T)
        TV=self.dropout(TV)
        TV=self.FFN_TV(TV)

        VA=self.MHSCA_VA(V,A,attn_mask_VA)
        VA=self.layernorm(VA+V)
        VA=self.dropout(VA)
        VA=self.FFN_VA(VA)

        VT=self.MHSCA_VT(V,T,attn_mask_VT)
        VT=self.layernorm(VT+V)
        VT=self.dropout(VT)
        VT=self.FFN_VT(VT)

        AT=self.MHSCA_AT(A,T,attn_mask_AT)
        AT=self.layernorm(AT+A)
        AT=self.dropout(AT)
        AT=self.FFN_AT(AT)

        AV=self.MHSCA_AV(A,V,attn_mask_AV)
        AV=self.layernorm(AV+A)
        AV=self.dropout(AV)
        AV=self.FFN_AV(AV)

        #########prefusion
        #T=self.SA_T_ori(T,attn_mask_T_simple)
        #V = self.SA_V_ori(V, attn_mask_V_simple)
        #A = self.SA_A_ori(A, attn_mask_A_simple)
        #T_C=self.SA_T_context_ori(T_C,attn_mask_T_context_simple)
        #V_C=self.SA_V_context_ori(V_C,attn_mask_V_context_simple)
        #A_C=self.SA_A_context_ori(A_C,attn_mask_A_context_simple)


        #T=self.MHSA_T(T,attn_mask_T_self)
        #V=self.MHSA_V(V,attn_mask_V_self)
        #A=self.MHSA_A(A,attn_mask_A_self)

        #T = torch.reshape(T, (T.shape[0], -1))
        #V = torch.reshape(V, (V.shape[0], -1))
        #A = torch.reshape(A, (A.shape[0], -1))
        #TV = torch.reshape(TV, (TV.shape[0], -1))
        #TA = torch.reshape(TA, (TA.shape[0], -1))
        #VT = torch.reshape(VT, (VT.shape[0], -1))
        #VA = torch.reshape(VA, (VA.shape[0], -1))
        #AT = torch.reshape(AT, (AT.shape[0], -1))
        #AV = torch.reshape(AV, (AV.shape[0], -1))
        #VA = torch.reshape(VA, (VA.shape[0], -1))

        '''
        fusion方式：
        1：conv1d特征在特征维度与另外两个对应crossmodal输出特征拼接
        2：conv1d特征attention加权为长度1后在全连接层前与另外两个对应crossmodal加权为长度1的输出特征拼接
        3：使用原始特征
        1相对有效
        对输入数据先进行layer normalization有效果
        20epoch左右test acc到达峰值 71.83
        '''

        final_T=torch.cat((TA,TV,T),dim=2)
        #final_T=TA
        #final_T=torch.reshape(final_T,(final_T.shape[0],-1))
        final_T=self.SA_T(final_T,attn_mask_T_simple)
        #final_T=torch.cat((final_T,T_C),dim=1)


        final_V=torch.cat((VT,VA,V),dim=2)
        final_V=self.SA_V(final_V,attn_mask_V_simple)
        #final_V=torch.cat((final_V,V_C),dim=1)

        final_A=torch.cat((AT,AV,A),dim=2)
        final_A=self.SA_A(final_A,attn_mask_A_simple)
        #final_A=torch.cat((final_A,A_C),dim=1)
        #'''
        #######atmf#######
        #tensor of shape:(batch,3,3*dim_v)
        #tensor of shape:(batch,1,dim_v)
        final_T=torch.unsqueeze(final_T,1)
        final_V = torch.unsqueeze(final_V, 1)
        final_A = torch.unsqueeze(final_A, 1)

        Trimodal=torch.cat((final_T,final_V,final_A),dim=1)
        Trimodal=self.atmf(Trimodal)
        Trimodal=torch.reshape(Trimodal,(Trimodal.shape[0],-1))
        #'''
        Trimodal_C=self.Context(T_C,V_C,A_C)
        Trimodal=torch.cat((Trimodal,Trimodal_C),dim=1)

        #final=T+TV+TA
        final = self.fc1(Trimodal)
        #final = F.relu(final)
        final = self.bn1(final)
        final = F.dropout(final, p=0.5,training=self.training)
        final = self.fc2(final)
        #final = F.relu(final)
        final = self.bn2(final)
        final = F.dropout(final, p=0.5,training=self.training)
        final = self.fc3(final)
        #final = F.relu(final)
        final = self.bn3(final)
        final = F.dropout(final, p=0.5,training=self.training)
        final = self.fc4(final)

        return final

    # input: tensor of Q,K
    # output: tensor of shape:(q_len,k_len)
    def get_attention_padding_mask(self, q, k):
        attn_pad_mask = q[:, :, 0].eq(0).unsqueeze(1).repeat(1, k.size(1), 1)
        attn_pad_mask=attn_pad_mask.transpose(2,1)
        # |attn_pad_mask| : (batch_size, q_len, k_len)

        return attn_pad_mask

    def get_attention_padding_mask_simple(self, q):
        attn_pad_mask = q[:, :, 0].eq(0).unsqueeze(1)
        attn_pad_mask=attn_pad_mask.transpose(2,1)
        # |attn_pad_mask| : (batch_size, q_len, 1)

        return attn_pad_mask

    def get_attention_padding_mask_self(self, q):
        attn_pad_mask = q[:, :, 0].eq(0).unsqueeze(1).repeat(1, q.size(1), 1)
        attn_pad_mask=attn_pad_mask.transpose(2,1)
        # |attn_pad_mask| : (batch_size, q_len, q_len)

        return attn_pad_mask

    def atmf(self, mm_feature):

        s = self.atmf_dense_4(self.atmf_dense_3(self.atmf_dense_2(self.atmf_dense_1(mm_feature))))
        s = s.permute(0, 2, 1)

        s = F.softmax(s, dim=2) + 1

        mm_feature=mm_feature.permute(0,2,1)

        wei_fea = mm_feature * s
        output=wei_fea.permute(0,1,2)
        batch_size=mm_feature.shape[0]
        assert batch_size==self.W_F[:batch_size,:,:].shape[0]
        P_F = torch.tanh(torch.bmm(self.W_F[:batch_size,:,:], wei_fea))
        P_F = F.softmax(P_F, dim=2)

        gamma_f = torch.bmm(self.W_f[:batch_size,:,:].permute(0, 2, 1), P_F)
        gamma_f = gamma_f.permute(0, 2, 1)
        atmf_output = torch.bmm(wei_fea, gamma_f)
        atmf_output=atmf_output.permute(0,2,1)

        return output


class Trimodal_CMD(nn.Module):
    def __init__(self,length_x,length_y,length_z,dim_in_x,dim_in_y,dim_in_z,dim_k,dim_v):
        super(Trimodal_CMD, self).__init__()
        self.n_T=length_x
        self.n_V=length_y
        self.n_A=length_z
        self.dim_T=dim_in_x
        self.dim_V=dim_in_y
        self.dim_A=dim_in_z
        self.dim_k=dim_k
        self.dim_v=dim_v
        ###########Attention######
        self.MHSA=MultiHeadSelfAttention(3,dim_k,dim_k,dim_k)
        self.MHSCA_TV=MultiHeadCrossSelfAttention(self.n_T,self.n_V,self.dim_k,self.dim_k,self.dim_k,self.dim_v)
        self.MHSCA_TA=MultiHeadCrossSelfAttention(self.n_T,self.n_A,self.dim_k,self.dim_k,self.dim_k,self.dim_v)
        self.MHSCTA_TV = MultiHeadContrastiveSelfAttention(self.n_T, self.n_V, self.dim_k, self.dim_k, self.dim_k, self.dim_v)
        self.MHSCTA_TA = MultiHeadContrastiveSelfAttention(self.n_T, self.n_A, self.dim_k, self.dim_k, self.dim_k,
                                                           self.dim_v)
        self.MHSCA_VA = MultiHeadCrossSelfAttention(self.n_V, self.n_A, self.dim_k, self.dim_k, self.dim_k, self.dim_v)
        self.MHSCA_AT = MultiHeadCrossSelfAttention(self.n_A, self.n_T, self.dim_k, self.dim_k, self.dim_k, self.dim_v)
        self.MHSCA_AV = MultiHeadCrossSelfAttention(self.n_A, self.n_V, self.dim_k, self.dim_k, self.dim_k, self.dim_v)
        self.MHSCA_VT = MultiHeadCrossSelfAttention(self.n_V, self.n_T, self.dim_k, self.dim_k, self.dim_k, self.dim_v)
        ###########

        ##########same size encoder
        self.conv_T=nn.Conv1d(dim_in_x,dim_v,kernel_size=1)
        self.conv_V = nn.Conv1d(dim_in_y, dim_v, kernel_size=1)
        self.conv_A = nn.Conv1d(dim_in_z, dim_v, kernel_size=1)
        self.proj_T=nn.Linear(dim_in_x,dim_v)
        self.proj_V = nn.Linear(dim_in_y, dim_v)
        self.proj_A = nn.Linear(dim_in_z, dim_v)
        self.dropout=nn.Dropout(p=0.5)
        self.layernorm=nn.LayerNorm(dim_v)
        self.layernorm_T = nn.LayerNorm(self.dim_T)
        self.layernorm_V = nn.LayerNorm(self.dim_V)
        self.layernorm_A = nn.LayerNorm(self.dim_A)

        ########shared encoder
        #self.shared=nn.Linear(dim_v,dim_v)
        self.shared = nn.Sequential()
        self.shared.add_module('shared_1', nn.Linear(in_features=dim_v, out_features=dim_v))

        self.linear_T=nn.Linear(self.dim_T,dim_v)

        ###########classify
        self.fc_dim1 = dim_v *3
        self.fc_dim2 = 192
        self.fc_dim3 = 32
        self.fc_dim4 = 8
        self.fc_dim5 = 2
        self.fc1 = nn.Linear(self.fc_dim1, self.fc_dim3)
        self.fc2 = nn.Linear(self.fc_dim3, self.fc_dim5)
        self.fc3 = nn.Linear(self.fc_dim3, self.fc_dim4)
        self.fc4 = nn.Linear(self.fc_dim4, self.fc_dim5)
        self.bn1 = nn.BatchNorm1d(num_features=self.fc_dim3, eps=1e-05, momentum=0.1, affine=True,
                                  track_running_stats=True)
        self.bn2 = nn.BatchNorm1d(num_features=self.fc_dim3, eps=1e-05, momentum=0.1, affine=True,
                                  track_running_stats=True)
        self.bn3 = nn.BatchNorm1d(num_features=self.fc_dim4, eps=1e-05, momentum=0.1, affine=True,
                                  track_running_stats=True)

    def forward(self, T, V,A):

        #T=self.layernorm_T(T)
        #V = self.layernorm_V(V)
        #A = self.layernorm_A(A)
        T=T.transpose(1,2)
        V=V.transpose(1, 2)
        A=A.transpose(1, 2)

        T=self.proj_T(T)
        V=self.proj_V(V)
        A=self.proj_A(A)



        T=self.layernorm(T)
        T=self.dropout(T)
        V=self.layernorm(V)
        V=self.dropout(V)
        A=self.layernorm(A)
        A=self.dropout(A)

        self.T_shared=self.shared(T)
        self.V_shared=self.shared(V)
        self.A_shared=self.shared(A)

        TVA=torch.cat((self.T_shared,self.V_shared,self.A_shared),dim=1)
        residual=TVA
        TVA=self.MHSA(TVA)
        TVA=self.layernorm(TVA+residual)
        TVA=self.dropout(TVA)




        TVA = torch.reshape(TVA, (TVA.shape[0], -1))



        #final=T+TV+TA
        final = self.fc1(TVA)
        #att = F.relu(att)
        final = self.bn1(final)
        final = F.dropout(final, p=0.5,training=self.training)
        final = self.fc2(final)



        return final