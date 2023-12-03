#!usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, KBinsDiscretizer, LabelEncoder
from sklearn.model_selection import train_test_split

col_names = ['id', 'click', 'hour', 'C1', 'banner_pos', 'site_id', 'site_domain', 'site_category', 'app_id', 'app_domain',
         'app_category', 'device_id', 'device_ip', 'device_model', 'device_type', 'device_conn_type', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19',
         'C20', 'C21']


def data_process(file: None, nrows):
    data_df = pd.read_csv(file, names=col_names, nrows=nrows, header=0)
    features = col_names[2:]
    print(features)
    label = ['click']
    data_df[features] = data_df[features].fillna('-1')
    del data_df['id']
    # data_df['hour'] = data_df['hour'] /
    # 对种类特征进行编码
    for feat in features:
        le = LabelEncoder()
        data_df[feat] = le.fit_transform(data_df[feat]) + 1
    # ord = OrdinalEncoder()
    # data_df[features] = ord.fit_transform(data_df[features

    df = pd.concat([pd.DataFrame(data_df)], axis=1, join='outer')
    print("data_process")
    print(df)
    # 划分验证集
    # train_set, val_set = train_test_split(df, test_size=sizes[

    # 保存文件
    df.reset_index(drop=True, inplace=True)
    # val_set.reset_index(drop=True, inplace=True)

    # print(train_set)
    df.to_csv('../project/dataset/avazu/data_5w.csv', index=0, header=0)
    # val_set.to_csv('../project/dataset/avazu/val_1000w.csv', index=0, header=0)


def user_action():
    import copy
    df = pd.read_csv("../project/dataset/avazu/data_5w.csv", header=None)
    action_map = dict()
    res = []
    for index, data in df.iterrows():
        if action_map.get(data[10]) is None:
            action_map[data[10]] = []
        res.append(copy.deepcopy(action_map[data[10]]))
        if data[0] == 1:
                # action_map[data[11]] = []
                action_map[data[10]] += [data[1], data[6]] # data[4], data[6],
                if len(action_map[data[10]]) > 40:
                    action_map[data[10]].__delitem__(0)
                    action_map[data[10]].__delitem__(0)
    # print(res)
    # for index in res:
    #     length = 80 - len(index)
    #     l = [0 for i in range(0, length)]
    #     index += l
    df2 = pd.DataFrame(res)

    print("user_action")
    print(df2)
    df = pd.concat([df, df2], axis=1, join='outer')#.fillna(0)

    print(df)
    df.reset_index(drop=True, inplace=True)
    df.to_csv('../project/dataset/avazu/data_5w.csv', index=0, header=0)


def split():
    data = pd.read_csv("../project/dataset/avazu/data_5w.csv", header=None)

    train_set, val_set = train_test_split(data, test_size=0.2, random_state=2023)
    train_set.reset_index(drop=True, inplace=True)
    val_set.reset_index(drop=True, inplace=True)
    # print(train_set)
    print("最终保存前的数据")
    print(train_set)
    print(val_set)
    train_set.to_csv("../project/dataset/avazu/train_5w.csv", index=0, header=0)
    val_set.to_csv("../project/dataset/avazu/val_5w.csv", index=0, header=0)


if __name__ == '__main__':
    data_process("../project/dataset/avazu/train.csv", nrows=50000)
    user_action()
    split()
    # from matplotlib import pyplot as plt
    # import seaborn as sns
    # sns.set()
    # data_df = pd.read_csv("../project/dataset/avazu/train.csv", names=col_names, nrows=5000000, header=0)
    # features = col_names[2:]
    # # label = ['click']
    # data_df[features] = data_df[features].fillna('-1')
    # del data_df['id']
    # # data_df['hour'] = data_df['hour'] /
    # # 对种类特征进行编码
    # for feat in features:
    #     le = LabelEncoder()
    #     data_df[feat] = le.fit_transform(data_df[feat]) + 1
    # plt.figure(figsize=(20, 18))
    # sns.heatmap(data_df.corr().abs(), annot=True)
    # plt.show()
    # file = '../project/dataset/avazu/train.csv'
    # nrows = 10000000
    # sizes = [0.75, 0.25]
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # data_process(file, nrows, sizes, device)

