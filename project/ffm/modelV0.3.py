import torch
from torch import nn


class FFM(nn.Module):
    def __init__(self, dense_feature_index, sparse_feature_index, emb_dim):
        super(FFM, self).__init__()
        self.dense_feature_index = dense_feature_index
        self.sparse_feature_index = sparse_feature_index
        self.emb_dim = emb_dim
        self.dense_feature_num = len(dense_feature_index)
        self.sparse_feature_num = len(sparse_feature_index)
        self.feature_num = self.dense_feature_num + self.sparse_feature_num * emb_dim
        self.field_num = self.dense_feature_num + self.sparse_feature_num
        # TODO 线性部分
        self.linear = nn.Linear(self.feature_num, 1)

        # TODO 二阶特征交叉部分
        # 二阶特征交叉矩阵
        self.field_matrix = nn.Parameter(torch.randn(self.feature_num, self.field_num, self.emb_dim))
        feature_to_field = dict()
        feature_to_field.update({1: 13})

    def feature_to_field(self, n):
        return n if n < 14 else (n - 14) // self.emb_dim + 14

    def forward(self, x):

        dense_feature = x[:, :13]
        sparse_feature = x[:, 13:].long()
        # 离散特征转化为嵌入向量
        result = [self.embedding_dict["embedding_" + str(i)](sparse_feature[:, i]) for i in
                  range(sparse_feature.shape[1])]
        sparse_feature = torch.cat(result, dim=1)
        # 合并特征 (batch_size, feature_num)
        all_feature = torch.cat([dense_feature, sparse_feature], dim=1)

        result = 0
        for i in range(1, self.feature_num):
            for j in range(i + 1, self.feature_num):
                result += torch.dot(self.field_matrix[i, self.feature_to_field(j)], self.field_matrix[j, self.feature_to_field(i)]) * torch.mul(all_feature[:, i], all_feature[:, j])
        output = self.linear(all_feature) + result
        output = output.reshape(-1)
        output = torch.sigmoid(output)
        return output.to(dtype=torch.float64)

m1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(m1)
print(m1[:, 0])




