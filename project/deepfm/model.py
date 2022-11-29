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


class FM(torch.nn.Module):
    def __init__(self, feature_number, k):
        """
        :param feature_number:  特征个数
        :param k:               矩阵分解的维度
        """
        super(FM, self).__init__()
        # 偏差
        self.w0 = torch.nn.Parameter(torch.rand([1, ]))
        # 普通线性部分w系数
        self.w1 = torch.nn.Parameter(torch.rand([feature_number, 1]))
        # 特征交叉部分
        self.w2 = torch.nn.Parameter(torch.rand([feature_number, k]))

    def forward(self, X):
        y1 = torch.mm(X, self.w1) + self.w0
        y2 = 1/2 * torch.sum(
            torch.pow(torch.mm(X, self.w2), 2) - torch.mm(torch.pow(X, 2), torch.pow(self.w2, 2)),
            dim=1,
            keepdim=True
        )
        # print(y1)
        # print(y2)
        return y1 + y2


class Dnn(nn.Module):
    """
    Dnn part
    """
    def __init__(self, hidden_units, dropout=0.):
        """
        hidden_units: 列表， 每个元素表示每一层的神经单元个数， 比如[256, 128, 64], 两层网络， 第一层神经单元128， 第二层64， 第一个维度是输入维度
        dropout: 失活率
        """
        super(Dnn, self).__init__()

        self.dnn_network = nn.ModuleList(
            [nn.Linear(layer[0], layer[1]) for layer in list(zip(hidden_units[:-1], hidden_units[1:]))])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        for linear in self.dnn_network:
            x = linear(x)
            x = F.relu(x)
        x = self.dropout(x)
        return x


class DeepFM(torch.nn.Module):
    def __init__(self, feature_num, deep_feature_input_nums, sparse_feature_nums, dnn_dropout=0.):
        super(DeepFM, self).__init__()
        self.fm = FM(feature_num, feature_num // 2)

        # 构造Embedding向量, 第一个参数是每个特征的样本数量
        self.embedding_dict = torch.nn.ModuleDict()
        for i, num in enumerate(sparse_feature_nums):
            self.embedding_dict.add_module("embedding_" + str(i), torch.nn.Embedding(num, 8))
        self.dnn = Dnn(deep_feature_input_nums)
        self.final_linear = torch.nn.Linear(64, 1)

    def forward(self, X):
        dense_feature = X[:, :13]
        sparse_feature = X[:, 13:].long()

        # result = []
        # for i in range(0, len(self.embedding_dict)):
        #     result.append(self.embedding_dict["embedding_" + str(i)](sparse_feature[:, i]))
        result = [self.embedding_dict["embedding_" + str(i)](sparse_feature[:, i]) for i in range(sparse_feature.shape[1])]

        sparse_feature = torch.cat(result, dim=1)
        all_feature = torch.cat([dense_feature, sparse_feature], dim=1)
        # Dnn输出
        y2 = self.final_linear(self.dnn(all_feature))
        # 因子分解机输出
        y1 = self.fm(all_feature)
        # print(y1/10000)
        # print(y2)
        return torch.reshape(torch.sigmoid(0.5 * (y1 + y2)), (1, -1)).squeeze(dim=0)


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
    hidden_units = [512, 256, 128, 64]
    dnn_dropout = 0.
    print(feature_columns)
    model = DeepFM(13 + 26 * 8, [26 * 8 + 13] + hidden_units, feature_columns, dnn_dropout).to(device=device)
    # model = WideDeep([index for index in range(1, 14)], [index for index in range(14, 40)], hidden_units, feature_columns, dnn_dropout).to(device=device)
    # summary(model, input_shape=(trn_x.shape[1],))

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
                loss = loss_func(predictions.cpu(), labels.cpu())
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
                val_loss = loss_func(predictions.cpu(), labels.cpu())
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

