
import torch.nn as nn
import torch
from FFM import FFM
from DNN import DNN


class DeepFFM(nn.Module):

    def __init__(self, sparse_feature_index, dense_feature_index, hidden_units, emb_size, sparse_feature_type, implicit_vector_dim, device, dnn_dropout=0.):
        """
        
        :param hidden_units: list(int)类型，为DNN中间层每层的节点个数
        """
        super(DeepFFM, self).__init__()
        self.sparse_feature_type = sparse_feature_type
        self.dnn_dropout = dnn_dropout
        self.emb_size = emb_size
        self.hidden_units = hidden_units
        self.dense_feature_index = dense_feature_index
        self.sparse_feature_index = sparse_feature_index
        self.implicit_vector_dim = implicit_vector_dim
        self.device = device

        self.field_num = len(dense_feature_index) + len(sparse_feature_index)
        self.sparse_feature_num = len(sparse_feature_index)
        self.dense_feature_num = len(dense_feature_index)
        self.feature_num = self.sparse_feature_num * self.emb_size + self.dense_feature_num

        # 构造Embedding向量, 第一个参数是每个特征的样本数量
        self.embedding_dict = torch.nn.ModuleDict()
        for i, num in enumerate(sparse_feature_type):
            self.embedding_dict.add_module("embedding_" + str(i), torch.nn.Embedding(num, self.emb_size))
        self.hidden_units = [self.feature_num] + self.hidden_units
        # DNN部分
        self.dnn = DNN(self.hidden_units)
        self.final_linear = torch.nn.Linear(hidden_units[-1], 1)

        # FFM部分
        self.ffm = FFM(self.feature_num, self.field_num, self.implicit_vector_dim, device)

    def forward(self, X):
        dense_feature = X[:, :13]
        sparse_feature = X[:, 13:].long()
        # 离散特征转化为嵌入向量
        result = [self.embedding_dict["embedding_" + str(i)](sparse_feature[:, i]) for i in
                  range(sparse_feature.shape[1])]
        sparse_feature = torch.cat(result, dim=1)
        # 合并特征 (batch_size, feature_num)
        all_feature = torch.cat([dense_feature, sparse_feature], dim=1)

        # Dnn输出
        y2 = self.final_linear(self.dnn(all_feature))
        # 因子分解机输出
        y1 = self.ffm(all_feature).reshape(-1, 1)
        # print(y1.size())
        # print(y2.size())
        return torch.reshape(torch.sigmoid(0.5 * (y1 + y2)), (1, -1)).squeeze(dim=0)

