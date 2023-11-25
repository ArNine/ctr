import torch
import torch.nn as nn
import torch.nn.functional as F
from informer.FFMS import FFMS
from informer.DNN import DNN
from Informer import Informer


class DeepFFMS(nn.Module):
    def __init__(self, feature_num, user_action_feature_num, feature_columns, dnn_dropout, emb_size, hidden_units, field_num, implicit_vector_dim, device):
        super(DeepFFMS, self).__init__()
        self.dnn_dropout = dnn_dropout
        self.emb_size = emb_size
        self.hidden_units = hidden_units
        self.implicit_vector_dim = implicit_vector_dim
        self.device = device
        self.feature_num = feature_num
        self.user_action_feature_num = user_action_feature_num
        self.field_num = field_num
        # 构造Embedding向量, 第一个参数是每个特征的样本数量
        self.embedding_dict = torch.nn.ModuleDict()
        for i, num in enumerate(feature_columns):
            self.embedding_dict.add_module("embedding_" + str(i), torch.nn.Embedding(num, self.emb_size))
        self.hidden_units = [self.feature_num * 8 + self.user_action_feature_num//8 * 16] + self.hidden_units
        # DNN部分
        self.dnn = DNN(self.hidden_units, F.leaky_relu, dnn_dropout)
        self.final_linear = torch.nn.Linear(hidden_units[-1], 1)
        # d_model, n_head, dropout
        self.informer = Informer(16, 2, 0.5)
        # FFM部分
        self.ffms = FFMS(self.feature_num * self.emb_size, self.field_num, self.implicit_vector_dim, device)

    def forward(self, x):

        sparse_feature = x[:, :self.feature_num].long()
        action_feature = x[:, self.feature_num:].long()
        # 离散特征转化为嵌入向量
        result = [self.embedding_dict["embedding_" + str(i)](sparse_feature[:, i]) for i in
                  range(sparse_feature.shape[1])]
        # 0 5
        res = []
        # print(action_feature.shape[1])
        for i in range(action_feature.shape[1]//2):
            m1 = self.embedding_dict["embedding_" + str(0)](action_feature[:, i * 2])
            m2 = self.embedding_dict["embedding_" + str(5)](action_feature[:, i * 2 + 1])
            res.append(torch.cat([m1, m2], dim=1))
        # action_feature = [self.embedding_dict["embedding_" + str(0 if i == 0 else 5)](action_feature[:, i]) for i in
        #  range(action_feature.shape[1])]
        action_feature = torch.stack(res, dim=1)

        action_feature = self.informer(action_feature)

        sparse_feature = torch.cat(result, dim=1)
        # print(action_feature.shape)
        action_feature = action_feature.reshape([-1, 80])
        # print(action_feature.shape)
        # 合并特征 (batch_size, feature_num)
        all_feature = torch.cat([sparse_feature, action_feature], dim=1)

        # Dnn输出
        y2 = self.final_linear(self.dnn(all_feature))
        # 因子分解机输出
        y1 = self.ffms(sparse_feature).reshape(-1, 1)
        # print(y1.size())
        # print(y2.size())
        return torch.reshape(torch.sigmoid(0.5 * (y1 + y2)), (1, -1)).squeeze(dim=0)




