#!usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, KBinsDiscretizer
from sklearn.model_selection import train_test_split


class DataProcess():
    def __init__(self, file, nrows, sizes, device):
        # 特征列名
        names = ['label', 'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11',
                 'I12', 'I13', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',
                 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22',
                 'C23', 'C24', 'C25', 'C26']
        self.device = device
        # 读取数据
        self.data_df = pd.read_csv(file, sep="\t", names=names, nrows=nrows)
        self.data = self.feature_process()

    def feature_process(self):
        # 连续特征
        dense_features = ['I' + str(i) for i in range(1, 14)]
        # 离散特征
        sparse_features = ['C' + str(i) for i in range(1, 27)]
        features = dense_features + sparse_features

        # 缺失值填充:连续特征缺失值填充0；离散特征缺失值填充'-1'
        self.data_df[dense_features] = self.data_df[dense_features].fillna(0)
        self.data_df[sparse_features] = self.data_df[sparse_features].fillna('-1')

        # 连续特征等间隔分箱
        kb = KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='uniform')
        self.data_df[dense_features] = kb.fit_transform(self.data_df[dense_features])

        # 特征进行连续编码，为了在与参数计算时使用索引的方式计算，而不是向量乘积
        ord = OrdinalEncoder()
        self.data_df[features] = ord.fit_transform(self.data_df[features])

        self.data = self.data_df[features + ['label']].values
        return self.data

    def train_valid_test_split(self, sizes):
        train_size, test_size = sizes[0], sizes[1]

        # 每一列的最大值加1
        field_dims = (self.data.max(axis=0).astype(int) + 1).tolist()[:-1]

        # 数据集分割为训练集、验证集、测试集
        train_data, test_data = train_test_split(self.data, train_size=train_size, random_state=2022)

        # 将ndarray格式转为tensor格式
        x_train = torch.tensor(train_data[:, :-1], dtype=torch.long).to(self.device)
        y_train = torch.tensor(train_data[:, -1], dtype=torch.float32).to(self.device)
        x_test = torch.tensor(test_data[:, :-1], dtype=torch.long).to(self.device)
        y_test = torch.tensor(test_data[:, -1], dtype=torch.float32).to(self.device)

        return field_dims, (x_train, y_train), (x_test, y_test)


if __name__ == '__main__':
    file = 'criteo-100k.txt'
    nrows = 100000
    sizes = [0.75, 0.25]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataprocess = DataProcess(file, nrows, sizes, device)
    field_dims, (x_train, y_train), (x_test, y_test) \
        = dataprocess.train_valid_test_split(sizes)
    print(x_train.shape)
    print(field_dims)
    offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
    print(offsets)

