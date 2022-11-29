#!usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import numpy as np
import torch.nn as nn
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from torch import optim
import tqdm
from sklearn.metrics import f1_score, recall_score, roc_auc_score


class Feature_Embedding(nn.Module):
    def __init__(self, field_dims, emb_size):
        """
        :param field_dims: 特征数量列表，其和为总特征数量
        :param emb_size: embedding的维度
        """
        super(Feature_Embedding, self).__init__()
        # embedding层
        self.emb = nn.Embedding(sum(field_dims), emb_size)
        # 模型初始化
        nn.init.xavier_uniform_(self.emb.weight.data)
        # 偏置项
        self.offset = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

    def forward(self, x):
        # self.offset中存储的是每一列特征计数的开始值
        # x + x.new_tensor(self.offset)：x中的每一列是分别进行顺序编码+起始值后就可以在self.emb中找到真正的索引
        x = x + x.new_tensor(self.offset)
        return self.emb(x.long())


class FieldAwareFactorizationMachine(nn.Module):

    def __init__(self, field_dims, embed_dim=4):
        """
        :param field_dims: field_dims列表存储的是每一列有多少种特征
        :param embed_dim: 向量维度
        """
        super(FieldAwareFactorizationMachine, self).__init__()
        self.field_dims = field_dims
        # 一阶偏置项
        self.bias = nn.Parameter(torch.zeros((1,)))
        # 一阶部分
        self.embed_linear = Feature_Embedding(field_dims, 1)
        # 二阶部分,self.embed_cross形式为[num_fields,sum(field_dims),embed_size]
        self.embed_cross = nn.ModuleList([Feature_Embedding(field_dims, embed_dim) for _ in field_dims])

    def forward(self, x):
        """
        前向传播
        :param x: 输入数据
        :return:
        """
        # x shape: (batch_size, num_fields)
        # embed(x) shape: (batch_size, num_fields, embed_dim)
        # 特征域的个数
        num_fields = len(self.field_dims)
        # 列表中有num_fields个(batch_size, num_fields, embed_dim)
        embeddings = [embed(x) for embed in self.embed_cross]
        # (batch_size, num_fields*num_fields, embed_dim)
        embeddings = torch.cat(embeddings, dim=1)

        # 分别表示f_j,f_i
        i1, i2 = [], []
        for i in range(num_fields):
            for j in range(i + 1, num_fields):
                i1.append(j * num_fields + i)   # f_j
                i2.append(i * num_fields + j)   # f_i

        # 将w_{i,f_j}x_i与w_{j,f_j}x_j对应位置进行点乘
        # (batch,1)
        embedding_cross = torch.mul(embeddings[:, i1], embeddings[:, i2]).sum(dim=2).sum(dim=1, keepdim=True)
        # (batch,1)
        output = self.embed_linear(x).sum(dim=1) + self.bias + embedding_cross
        output = torch.sigmoid(output)
        return output


FILE_PATH = "../dataset/cirte_small"


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


nrows = 100000
learning_rate = 1e-4
weight_decay = 1e-6


def train_and_test(train_dataloader, test_dataloader, model, device, num_epochs=100):
    # 损失函数
    criterion = nn.BCELoss()
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # 记录训练与测试过程的损失，用于绘图
    train_loss, test_loss, train_acc, test_acc = [], [], [], []
    for epoch in range(num_epochs):
        train_loss_sum = 0.0
        train_len = 0
        train_correct = 0
        # 显示训练进度
        train_dataloader = tqdm.tqdm(train_dataloader)
        train_dataloader.set_description('[%s%04d/%04d]' % ('Epoch:', epoch + 1, num_epochs))

        # 训练模式
        model.train()
        model.to(device)
        for i, data_ in enumerate(train_dataloader):
            x, y = data_[0].to(device), data_[1].to(device)
            # 开始当前批次训练时，优化器的梯度置零，否则，梯度会累加
            optimizer.zero_grad()
            # output size = (batch,)
            output = model(x)
            # print(output)
            # print(output.size())
            loss = criterion(output, y.double())
            # 反向传播
            loss.backward()
            # 利用优化器更新参数
            optimizer.step()
            # BCELoss默认reduction="mean",因此需要乘以个数
            train_loss_sum += loss.detach() * len(x)
            train_len += len(y)
            _, predicted = torch.max(output, 1)
            train_correct += (predicted == y).sum().item()
            # print("train_correct=\n", train_correct)
            # print("train_acc=\n", train_correct / train_len)
            F1 = f1_score(y.cpu(), predicted.cpu(), average="weighted")
            Recall = recall_score(y.cpu(), predicted.cpu(), average="micro")

            # 设置日志
            postfic = {"train_loss: {:.5f},train_acc:{:.3f}%,F1: {:.3f}%,Recall:{:.3f}%".
                           format(train_loss_sum / train_len, 100 * train_correct / train_len, 100 * F1, 100 * Recall)}
            train_dataloader.set_postfix(log=postfic)
        train_loss.append((train_loss_sum / train_len).item())
        train_acc.append(round(train_correct / train_len, 4))

        # 测试
        test_dataloader = tqdm.tqdm(test_dataloader)
        test_dataloader.set_description('[%s%04d/%04d]' % ('Epoch:', epoch + 1, num_epochs))
        model.eval()
        model.to(device)
        with torch.no_grad():
            test_loss_sum = 0.0
            test_len = 0
            test_correct = 0
            for i, data_ in enumerate(test_dataloader):
                x, y = data_[0].to(device), data_[1].to(device)
                output = model(x)
                loss = criterion(output.squeeze(1), y)
                test_loss_sum += loss.detach() * len(x)
                test_len += len(y)
                _, predicted = torch.max(output, 1)
                test_correct += (predicted == y).sum().item()
                F1 = f1_score(y.cpu(), predicted.cpu(), average="weighted")
                Recall = recall_score(y.cpu(), predicted.cpu(), average="micro")
                # 设置日志
                postfic = {"test_loss: {:.5f},test_acc:{:.3f}%,F1: {:.3f}%,Recall:{:.3f}%".
                               format(test_loss_sum / test_len, 100 * test_correct / test_len, 100 * F1, 100 * Recall)}
                test_dataloader.set_postfix(log=postfic)
            test_loss.append((test_loss_sum / test_len).item())
            test_acc.append(round(test_correct / test_len, 4))

    return train_loss, test_loss, train_acc, test_acc


class FFM(nn.Module):
    def __init__(self, dense_feature_index, sparse_feature_index, feature_columns, emb_dim):
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
            self.embedding_dict.add_module("embedding_" + str(i), torch.nn.Embedding(num, 8))
        # TODO 线性部分
        self.linear = nn.Linear(self.feature_num, 1)


        # TODO 二阶特征交叉部分
        # 二阶特征交叉矩阵
        self.field_matrix = nn.Parameter(torch.randn(self.feature_num, self.field_num, self.emb_dim))

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

        result = 0
        for i in range(0, self.feature_num):
            for j in range(i + 1, self.feature_num):
                result += torch.dot(self.field_matrix[i, self.feature_to_field(j)], self.field_matrix[j, self.feature_to_field(i)]) * torch.mul(all_feature[:, i], all_feature[:, j])
        linear = self.linear(all_feature)

        output = linear + result.reshape([-1, 1])
        output = output.reshape(-1)
        output = torch.sigmoid(output)
        return output.to(dtype=torch.double)


def main():
    num_epochs = 100
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    # 把数据构建成数据管道,
    train_x, train_y, val_x, val_y, test, feature_columns = read_data(FILE_PATH)
    train_dataset = TensorDataset(torch.tensor(train_x, device=device).float(),
                                  torch.tensor(train_y, device=device).float())
    val_dataset = TensorDataset(torch.tensor(val_x, device=device).float(), torch.tensor(val_y, device=device).float())

    train = DataLoader(train_dataset, shuffle=True, batch_size=2048)
    val = DataLoader(val_dataset, shuffle=True, batch_size=2048)
    model = FFM([i for i in range(0, 13)], [i for i in range(13, 39)], feature_columns, 8)

    # 训练与测试
    train_loss, test_loss, train_acc, test_acc = train_and_test(train, val, model, device, num_epochs)

    # 绘图，展示损失变化
    epochs = np.arange(num_epochs)
    plt.plot(epochs, train_loss, 'b-', label='Training loss')
    plt.plot(epochs, test_loss, 'r--', label='Validation loss')
    plt.title('Training And Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    epochs = np.arange(num_epochs)
    plt.plot(epochs, train_acc, 'b-', label='Training acc')
    plt.plot(epochs, test_acc, 'r--', label='Validation acc')
    plt.title('Training And Validation acc')
    plt.xlabel('Epochs')
    plt.ylabel('acc')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()

