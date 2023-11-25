import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=1,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class Informer(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super(Informer, self).__init__()
        self.attn0 = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.attn1 = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.attn2 = nn.MultiheadAttention(d_model, n_head, dropout=dropout)

        self.conv0 = ConvLayer(d_model)
        self.conv1 = ConvLayer(d_model)

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # 残差
        res_x = x
        x = self.layer_norm(x)
        x, attn = self.attn0(x, x, x)
        x = res_x + x
        x = self.conv0(x)
        # print(x.shape)
        x, attn = self.attn1(x, x, x)
        x = self.conv1(x)
        x, attn = self.attn2(x, x, x)
        return x
