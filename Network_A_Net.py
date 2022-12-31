from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def forward(self, x):
        # x: tensor of shape (batch, n, dim_in)
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        nh = self.num_heads
        dk = self.dim_k // nh  # dim_k of each head
        dv = self.dim_v // nh  # dim_v of each head

        q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        v = self.linear_v(x).reshape(batch, n, nh, dv).transpose(1, 2)  # (batch, nh, n, dv)

        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact  # batch, nh, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, nh, n, n
        dist=self.dropout(dist)
        att = torch.matmul(dist, v)  # batch, nh, n, dv
        att = att.transpose(1, 2).reshape(batch, n, self.dim_v)  # batch, n, dim_v
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
        return result

class MultiHeadContrastiveSelfAttention(nn.Module):
    dim_in: int  # input dimension
    dim_k: int  # key and query dimension
    dim_v: int  # value dimension
    num_heads: int  # number of heads, for each head, dim_* = dim_* // num_heads

    def __init__(self, length_x,length_y,dim_in_x,dim_in_y, dim_k, dim_v, num_heads=8):
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
        dist = torch.softmax(1-dist, dim=-1)
        att = q+torch.matmul(dist, v)  # batch, nh, n, dv
        att = att.transpose(1, 2).reshape(batch, n_x, self.dim_v)  # batch, n, dim_v
        att = torch.reshape(att, (att.shape[0], -1))
        result=att
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
        return result

#Modality x->y
class CrossAttentionNetwork(nn.Module):
    def __init__(self,length_x,length_y,dim_in_x,dim_in_y,dim_k,dim_v):
        super(CrossAttentionNetwork, self).__init__()
        self.dim_in_x=dim_in_x
        self.dim_in_y=dim_in_y
        self.dim_k=dim_k
        self.dim_v=dim_v
        self.MHSA=MultiHeadSelfAttention(length_y,dim_k,dim_k,dim_v)
        self.MHCSA=MultiHeadCrossSelfAttention(length_x,length_y,dim_k,dim_k,dim_k,dim_v)
        self.MHSAL_x = MultiHeadSelfAttentionLayers(length_x, dim_in_x, dim_k, dim_v)
        self.MHSAL_y=MultiHeadSelfAttentionLayers(length_y, dim_in_y, dim_k, dim_v)
        self.layer_norm = nn.LayerNorm(dim_k)
        self.dropout = nn.Dropout(0.5)
        self.conv_x = nn.Conv1d(dim_in_x, dim_k, kernel_size=1)
        self.conv_y=nn.Conv1d(dim_in_y,dim_k,kernel_size=1)
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
        #result=self.MultiHeadCrossSelfAttention
        #MHCSA=self.MultiHeadCrossSelfAttention
        #MHSAL_x=self.MHSAL_x
        #MHSAL_y=self.MHSAL_y
        #x=MHSAL_x(x)
        #y=MHSAL_y(y)
        #att=result(x,y)
        #att=MHCSA(x,y)
        x=x.transpose(1,2)
        x=self.conv_x(x)
        #x=F.relu(x)
        #x=self.dropout(x)
        x=x.transpose(1,2)
        residual=x
        y=y.transpose(1,2)
        y=self.conv_y(y)
        #x=F.relu(x)
        #x=self.dropout(x)
        y=y.transpose(1,2)
        x=self.layer_norm(x)
        y=self.layer_norm(y)
        x=self.dropout(x)
        y=self.dropout(y)
        residual=x
        att=self.MHCSA(x,y)
        att=self.layer_norm(att)
        att=self.dropout(att)
        residul=att
        att=self.MHSA(att)
        att=self.layer_norm(att+residual)
        att=self.dropout(att)
        #att=self.dropout(att)
        att = torch.reshape(att, (att.shape[0], -1))
        att = self.fc1(att)
        att = F.relu(att)
        att = self.bn1(att)
        att = F.dropout(att, p=0.5,training=self.training)
        att = self.fc2(att)
        att = F.relu(att)
        att = self.bn2(att)
        att = F.dropout(att, p=0.5,training=self.training)
        att = self.fc3(att)
        att = F.relu(att)
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
        self.MHSA1 = MultiHeadSelfAttention(length, dim_in, dim_k, dim_v)
        self.MHSA2 = MultiHeadSelfAttention(length, dim_v, dim_k, dim_v)
        self.MHSA3 = MultiHeadSelfAttention(length, dim_v, dim_k, dim_v)
        self.dropout=nn.Dropout(p=0.5)
        self.layer_norm=nn.LayerNorm(dim_k)
        self.conv_A = nn.Conv1d(dim_in, dim_k, kernel_size=1)
        self.conv_1=nn.Conv1d(dim_in, 512, kernel_size=1)
        self.conv_2=nn.Conv1d(dim_in, 256, kernel_size=1)
        self.conv_3=nn.Conv1d(dim_in, 150, kernel_size=1)
        self.fc_dim1=dim_v*length #sequnece length
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
        self.bn1 = nn.BatchNorm1d(num_features=self.fc_dim2, eps=1e-05, momentum=0.1, affine=True,
                                 track_running_stats=True)
        self.bn2 = nn.BatchNorm1d(num_features=self.fc_dim3, eps=1e-05, momentum=0.1, affine=True,
                                 track_running_stats=True)
        self.bn3 = nn.BatchNorm1d(num_features=self.fc_dim4, eps=1e-05, momentum=0.1, affine=True,
                                 track_running_stats=True)

    def forward(self, x):
        # x: tensor of shape (batch, n, dim_in)
        #x=self.MHSA1(x)
        #x=self.layer_norm(x)
        #x=self.dropout(x)
        #residual=x
        x=x.transpose(1,2)
        x=self.conv_A(x)
        #x=F.relu(x)
        #x=self.dropout(x)
        x=x.transpose(1,2)
        residual=x
        #gru
        x,_=self.gru(x)
        residual=x
        #Att Layer
        x=self.MHSA2(x)
        x=self.layer_norm(x+residual)
        x=self.dropout(x)
        residual = x
        #FFN
        #x=self.w2(self.dropout(F.relu(self.w1(x))))
        #x=self.layer_norm(x+residual)
        #x=self.dropout(x)
        #Att Layer
        x=self.MHSA3(x)
        x=self.layer_norm(x+residual)
        x=self.dropout(x)
        result=x
        #FFN
        #x=self.w2(self.dropout(F.relu(self.w1(x))))
        #x=self.layer_norm(x+residual)
        #x=self.dropout(x)
        #FC
        x = torch.reshape(x, (x.shape[0], -1))
        x = self.fc1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.bn3(x)
        x = self.dropout(x)
        x = self.fc4(x)
        return x
