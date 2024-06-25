import torch.nn as nn
import torch
import torch.nn.functional as F
class DNN(nn.Module):
    """
    Dnn partxxxxxx
    """
    def __init__(self, hidden_units, active_func, dropout=0.4):
        """
        hidden_units: 列表， 每个元素表示每一层的神经单元个数， 比如[256, 128, 64], 两层网络， 第一层神经单元128， 第二层64， 第一个维度是输入维度
        active_func: 激活函数
        dropout: 失活率
        """
        super(DNN, self).__init__()
        # print(hidden_units)
        self.dnn_network = nn.ModuleList(
            [nn.Linear(layer[0], layer[1]) for layer in list(zip(hidden_units[:-1], hidden_units[1:]))])
        self.dropout = nn.Dropout(p=dropout)
        self.active_func = active_func

    def forward(self, x):
        for linear in self.dnn_network:
            x = linear(x)
            x = self.active_func(x)
        x = self.dropout(x)
        return x