import torch

import pandas
import numpy
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
FILE_PATH = "../dataset/cirte_small"


def to_csv():
    with open(f"{FILE_PATH}/train_1m.txt") as f:
        all = f.read().split("\n")
        dataset = [line.split("\t") for line in all]
        df = pandas.DataFrame(dataset)
        # 切分特征，dense为连续特征，sparse为离散特征
        dense = df.iloc[:, 0:14].replace('', -1)
        sparse = df.iloc[:, 14:].replace('', '0').fillna('0')
        # print(df)
        label = dense[0]
        dense = dense.iloc[:, 1:14]

        # 对种类特征进行编码
        for feat in sparse:
            le = LabelEncoder()
            sparse[feat] = le.fit_transform(sparse[feat])
            # print(sparse[feat])
        # 对连续特征进行归一化处理
        mms = MinMaxScaler()
        dense = mms.fit_transform(dense)

        # 分开测试集和训练集
        df = pandas.concat([pandas.DataFrame(label), pandas.DataFrame(dense), pandas.DataFrame(sparse)], axis=1, join='outer')
        # print(df[0])
        # train = df[:train_df.shape[0]]
        # test = data_df[train_df.shape[0]:]

        # train['Label'] = label

        # 划分验证集
        train_set, val_set = train_test_split(df, test_size=0.4, random_state=2020)
        val_set, test_set = train_test_split(val_set, test_size=0.5, random_state=2022)
        # 保存文件
        train_set.reset_index(drop=True, inplace=True)
        val_set.reset_index(drop=True, inplace=True)
        test_set.reset_index(drop=True, inplace=True)
        # print(train_set)
        train_set.to_csv(f'{FILE_PATH}/train_set.csv', index=0, header=0)
        val_set.to_csv(f'{FILE_PATH}/val_set.csv', index=0, header=0)
        test_set.to_csv(f'{FILE_PATH}/test_set.csv', index=0, header=0)

def process_data():
    dataset = pandas.read_csv("../dataset/cirte_small/train_1m.txt")
    print(dataset[0])


to_csv()
