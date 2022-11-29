import torch
from torch import nn


class FFMLayer(nn.Module):
    def __init__(self, feature_columns, k, w_reg=1e-4, v_reg=1e-4):
        super(FFMLayer, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.k = k  # 隐向量v的维度
        self.w_reg = w_reg  # 一阶权重w的正则化项
        self.v_reg = v_reg  # 二阶组合特征权重的正则化项

        # 真实的特征维度是：类别型变量做了one hot之后的维度加连续型变量的维度
        self.feature_num = sum([feat['feat_onehot_dim'] for feat in self.sparse_feature_columns]) + len(
            self.dense_feature_columns)
        # 域个数是原始特征的个数，一个特征属于一个域
        self.field_num = len(self.dense_feature_columns) + len(self.sparse_feature_columns)

        # 一阶线性部分
        self.linear = nn.Linear(self.feature_num, 1)
        self.v = nn.Parameter(torch.randn(self.feature_num, self.field_num, k))  # 二阶特征组合的交互矩阵

    def ffm_layer(self, inputs):
        # x的维度是(batch_size, 26):离散特征个数加连续特征个数，离散特征还没有做Onehot
        dense_input = inputs[:, :13]
        sparse_inputs = inputs[:, 13:]

        # 做One hot编码 将连续特征和one hot后的特征拼接成为每个样本新的特征
        x = dense_input.to(dtype=torch.float32)
        for i in range(sparse_inputs.shape[1]):
            one_hot_value = F.one_hot(sparse_inputs[:, i].to(dtype=torch.int64),
                                      num_classes=int(self.sparse_feature_columns[i]['feat_onehot_dim']))
            x = torch.cat([x, one_hot_value.to(dtype=torch.float32)], 1)
        linear_part = self.linear(x)
        inter_part = 0
        # 每维特征先跟自己的[field_num, k]相乘得到Vij*X  [batch_size, 2291] x [2291, 39, 8] = [batch_size, 39, 8]
        field_f = torch.tensordot(x, self.v, dims=1)
        # 域之间两两相乘
        for i in range(self.field_num):
            for j in range(i + 1, self.field_num):
                inter_part += torch.sum(torch.mul(field_f[:, i], field_f[:, j]), 1, keepdims=True)
        output = linear_part + inter_part
        output = output.reshape(-1)
        output = torch.sigmoid(output)
        return output.to(dtype=torch.float64)

    def forward(self, x):
        return self.ffm_layer(x)

    def fit(self, data, optimizer, epochs=100):
        # 训练模型并输出测试集每一轮的loss
        criterion = F.binary_cross_entropy
        for epoch in range(epochs):
            for t, (batch_x, batch_y) in enumerate(data):
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                total = self.forward(batch_x)
                loss = criterion(total, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loader_test = DataLoader(test_data, batch_size=10, shuffle=False)

            r = self.test(loader_test)
            print('Epoch %d, loss=%.4f' % (epoch, r))

    def test(self, data):
        # 测试集测试
        criterion = F.binary_cross_entropy
        all_loss = 0
        gt_labels = []
        pred_labels = []
        i = 0
        with torch.no_grad():
            for t, (batch_x, batch_y) in enumerate(data):
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                pred = self.forward(batch_x)
                gt_label = batch_y.cpu().data.numpy()
                pred_proba = pred.cpu().data.numpy()
                gt_labels.append(gt_label)
                pred_labels.append(pred_proba)
                loss = criterion(pred, batch_y)
                all_loss += loss.item()
                i += 1
        gt_labels, pred_labels = np.concatenate(gt_labels), np.concatenate(pred_labels)
        pred_labels = pred_labels.reshape(len(pred_labels), )
        auc = roc_auc_score(gt_labels, pred_labels)
        print('auc:', auc, 'gt_lables:', gt_labels.shape, 'pred_labels:', pred_labels.shape)
        return all_loss / i
