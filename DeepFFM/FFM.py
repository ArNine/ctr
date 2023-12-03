import torch.nn as nn
import torch


class FFM(nn.Module):
    def __init__(self, feature_num, field_num, implicit_vector_dim, device):
        super(FFM, self).__init__()
        self.implicit_vector_dim = implicit_vector_dim
        self.field_num = field_num
        self.feature_num = feature_num

        # 线性部
        self.linear = nn.Linear(self.feature_num, 1)
        # 二阶特征交叉部分, 二阶特征交叉矩阵
        self.field_matrix = nn.Parameter(torch.randn(self.feature_num, self.field_num, self.implicit_vector_dim))
        m1 = torch.randn([self.feature_num, self.field_num]).fill_(0)
        for i in range(0, self.feature_num):
            m1.__setitem__((i, self.feature_to_field(i)), 1)
        self.xor_matrix = m1.to(device)
        self.a = nn.Parameter(torch.randn([self.feature_num, self.feature_num]).float().fill_(1.0))

    def feature_to_field(self, n):
        # 13为dense_feature的个数
        return n if n < 13 else (n - 13) // self.implicit_vector_dim + 13

    def forward(self, X):
        all_feature = X
        all_feature_tran = all_feature.transpose(0, 1)
        result = 0
        for i in range(0, self.feature_num):
            # (field_num, emb_size)
            field_matrix = self.field_matrix[i]
            # (emb_size, field_num)
            m2 = field_matrix.transpose(0, 1)
            # (feature_num, field_num, field_num)
            m3 = torch.einsum('kj,ilk->ijl', [m2, self.field_matrix])
            # (feature_num, field_num)
            m4 = m3[:, self.feature_to_field(i)]
            # print(torch.mm(torch.mul(m4, self.xor_matrix).sum(dim=1).reshape(1, 221), all_feature_tran))
            result += torch.mul(torch.mm(torch.mul(m4, self.xor_matrix).sum(dim=1).reshape(1, self.feature_num), torch.einsum('kj,ki->ki', [self.a[i].reshape(self.feature_num, -1), all_feature_tran])),
                                all_feature[:, i])

        linear = self.linear(all_feature)
        output = linear + result.reshape([-1, 1])
        output = output.reshape(-1)
        output = torch.sigmoid(output)
        return output.to(dtype=torch.double)

