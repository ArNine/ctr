import datetime

import torch.nn as nn
import torch.nn.functional as F
import torch
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
# from torchkeras import summary, Model
FILE_PATH = "../dataset/cirte_small"


def plot_metric(dfhistory, metric):
    train_metrics = dfhistory[metric]
    val_metrics = dfhistory['val_' + metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation ' + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_" + metric, 'val_' + metric])
    plt.show()


def read_data(file_path):
    # 读入训练集，验证集和测试集
    train = pd.read_csv(file_path + '/train_set.csv')
    val = pd.read_csv(file_path + '/val_set.csv')
    test = pd.read_csv(file_path + '/test_set.csv')

    train.columns = [index for index in range(0, 40)]
    val.columns = [index for index in range(0, 40)]
    test.columns = [index for index in range(0, 40)]

    # 统计每种离散特征有多少枚举值
    all = pd.concat([train, val, test])
    # print(all)
    feature_columns = [all[index].max() + 1 for index in range(14, 40)]
    print("feature_columns:" + str(feature_columns))
    # print(train)
    train_x, train_y = train.drop(columns=0).values, train[0].values
    # print(train_y)
    val_x, val_y = val.drop(columns=0).values, val[0].values
    test = val.drop(columns=0).values

    return train_x, train_y, val_x, val_y, test, feature_columns


class FFM(nn.Module):
    def __init__(self, dense_feature_index, sparse_feature_index, feature_columns, emb_dim, device):
        super(FFM, self).__init__()
        self.dense_feature_index = dense_feature_index
        self.sparse_feature_index = sparse_feature_index
        self.emb_dim = emb_dim
        self.dense_feature_num = len(dense_feature_index)
        self.sparse_feature_num = len(sparse_feature_index)
        self.feature_num = self.dense_feature_num + self.sparse_feature_num * emb_dim
        self.field_num = self.dense_feature_num + self.sparse_feature_num
        self.embedding_dict = torch.nn.ModuleDict()
        for i, num in enumerate(feature_columns):
            self.embedding_dict.add_module("embedding_" + str(i), torch.nn.Embedding(num, self.emb_dim))
        # TODO 线性部分
        self.linear = nn.Linear(self.feature_num, 1)

        # TODO 二阶特征交叉部分
        # 二阶特征交叉矩阵
        self.field_matrix = nn.Parameter(torch.randn(self.feature_num, self.field_num, self.emb_dim))
        m1 = torch.randn([221, 39]).fill_(0)
        for i in range(0, 221):
            m1.__setitem__((i, self.feature_to_field(i)), 1)
        self.xor_matrix = m1.to(device)

    def feature_to_field(self, n):
        return n if n < 13 else (n - 13) // self.emb_dim + 13

    def forward(self, x):

        dense_feature = x[:, :13]
        sparse_feature = x[:, 13:].long()
        # 离散特征转化为嵌入向量
        result = [self.embedding_dict["embedding_" + str(i)](sparse_feature[:, i]) for i in
                  range(sparse_feature.shape[1])]
        sparse_feature = torch.cat(result, dim=1)
        # 合并特征 (batch_size, feature_num)
        all_feature = torch.cat([dense_feature, sparse_feature], dim=1)

        # 每维特征先跟自己的[field_num, k]相乘得到Vij*X  [batch_size, feature_num] x [feature_num, field_num, 8] = [batch_size, field_num, 8]

        # field_f = torch.tensordot(all_feature, self.field_matrix, dims=1)
        # # 域之间两两相乘
        # result = 0
        # for i in range(self.field_num):
        #     for j in range(i + 1, self.field_num):
        #         result += torch.sum(torch.mul(field_f[:, i], field_f[:, j]), dim=1, keepdims=True)


        # result = 0
        # for i in range(0, self.feature_num):
        #     for j in range(i + 1, self.feature_num):
        #         result += torch.mul(torch.dot(self.field_matrix[i, self.feature_to_field(j)], self.field_matrix[j, self.feature_to_field(i)]), torch.mul(all_feature[:, i], all_feature[:, j]))

        all_feature_tran = all_feature.transpose(0, 1)
        matrix = torch.mm(all_feature_tran, all_feature)
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
            # TODO matrix[i].sum()有问题，应该删除对角线以下元素,xor_matrix可能也不对
            # print(torch.mm(torch.mul(m4, self.xor_matrix).sum(dim=1).reshape(1, 221), all_feature_tran))
            result += torch.mul(torch.mm(torch.mul(m4, self.xor_matrix).sum(dim=1).reshape(1, 221), all_feature_tran), all_feature[:, i])
            # print(result.size())
            # print(result)

        linear = self.linear(all_feature)

        output = linear + result.reshape([-1, 1])
        output = output.reshape(-1)
        output = torch.sigmoid(output)
        # print(result)
        # print(output)
        return output.to(dtype=torch.double)

# 模型的相关设置
def auc(y_pred, y_true):
    pred = y_pred.data
    y = y_true.data
    return roc_auc_score(y, pred)


def run():
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    # 把数据构建成数据管道
    train_x, train_y, val_x, val_y, test, feature_columns = read_data(FILE_PATH)
    train_dataset = TensorDataset(torch.tensor(train_x, device=device).float(), torch.tensor(train_y, device=device).float())
    val_dataset = TensorDataset(torch.tensor(val_x, device=device).float(), torch.tensor(val_y, device=device).float())

    train = DataLoader(train_dataset, shuffle=True, batch_size=4096)
    val = DataLoader(val_dataset, shuffle=True, batch_size=4096)

    # 建立模型
    # hidden_units = [512, 256, 128, 64]
    # dnn_dropout = 0.
    print(feature_columns)
    model = FFM([i for i in range(0, 13)], [i for i in range(13, 39)], feature_columns, 8, device).to(device)

    print(model)
    loss_func = nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001, weight_decay=0.0005)
    metric_func = auc
    metric_name = 'auc'
    # 控制步长
    epochs = 100
    # 每几步打印下日志
    log_step_freq = 10
    dfhistory = pd.DataFrame(columns=['epoch', 'loss', metric_name, 'val_loss', 'val_' + metric_name])

    print('start_training.........')
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('========' * 8 + '%s' % nowtime)

    for epoch in range(1, epochs + 1):
        # 训练阶段
        model.train()
        loss_sum = 0.0
        metric_sum = 0.0
        step = 1
        for step, (features, labels) in enumerate(train, 1):
            # 梯度清零
            optimizer.zero_grad()
            # 正向传播
            predictions = model(features)
            # print(predictions)
            # print(labels)

            try:
                loss = loss_func(predictions.double().cpu(), labels.double().cpu())
            except:
                print(predictions.min())
                print(labels)

            try:
                metric = metric_func(predictions.cpu(), labels.cpu())
                # 反向传播
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
                metric_sum += metric.item()
            except ValueError:
                pass

            if step % log_step_freq == 0:
                print(("[step=%d] loss: %.3f, " + metric_name + ": %.3f") % (step, loss_sum / step, metric_sum / step))

        # 验证阶段
        model.eval()
        val_loss_sum = 0.0
        val_metric_sum = 0.0
        val_step = 1

        for val_step, (features, labels) in enumerate(val, 1):
            with torch.no_grad():
                predictions = model(features)
                val_loss = loss_func(predictions.double().cpu(), labels.double().cpu())
                try:
                    val_metric = metric_func(predictions.cpu(), labels.cpu())
                    val_loss_sum += val_loss.item()
                    val_metric_sum += val_metric.item()
                except ValueError:
                    pass

        # 记录日志
        info = (epoch, loss_sum / step, metric_sum / step, val_loss_sum / val_step, val_metric_sum / val_step)
        dfhistory.loc[epoch - 1] = info

        # 打印日志
        print(("\nEPOCH=%d, loss=%.3f, " + metric_name + " = %.3f, val_loss=%.3f, " + "val_" + metric_name + " = %.3f") % info)
        nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print('\n' + '==========' * 8 + '%s' % nowtime)

    print('Finished Training')
    print(dfhistory)

    # 观察损失和准确率的变化
    plot_metric(dfhistory, "loss")
    plot_metric(dfhistory, "auc")

    # 预测
    y_pred_probs = model(torch.tensor(test, device=device).float())
    y_pred = torch.where(y_pred_probs > 0.5, torch.ones_like(y_pred_probs), torch.zeros_like(y_pred_probs))
    print(y_pred.data)

if __name__ == '__main__':
    run()

