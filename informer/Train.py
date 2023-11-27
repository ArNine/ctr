
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import os
import datetime
import time
from DeepFFMS import DeepFFMS
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

class Log:
    def __init__(self, file_path):
        self.content = ""
        self.file_path = file_path

    def info(self, str_log):
        str_log = str(str_log)
        print(str_log)
        self.content += str_log + "\n"

    def save(self):
        with open(self.file_path, mode='w', encoding='utf-8') as f:
            f.write(self.content)


class Train:
    def __init__(self, dataset_file_path, feature_num, all_feature_num):
        super(Train, self).__init__()
        self.dataset_file_path = dataset_file_path
        self.feature_num = feature_num
        self.all_feature_num = all_feature_num

    def train(self):
        # 获取当前时间 yyyy-mm-dd hh:hour
        log_folder_name = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime())
        log = self.set_log(log_folder_name)
        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device("cuda")

        # 把数据构建成数据管道
        train_x, train_y, val_x, val_y, feature_columns = self.read_data()
        train_dataset = TensorDataset(torch.tensor(train_x, device=device).float(),
                                      torch.tensor(train_y, device=device).float())
        val_dataset = TensorDataset(torch.tensor(val_x, device=device).float(),
                                    torch.tensor(val_y, device=device).float())

        train = DataLoader(train_dataset, shuffle=True, batch_size=1500)
        val = DataLoader(val_dataset, shuffle=True, batch_size=1500)

        # 建立模型
        hidden_units = [256, 128, 64]
        dnn_dropout = 0.7
        emb_size = 8
        implicit_vector_dim = 8
        field_num = 22
        log.info(feature_columns)
        # feature_num, user_action_feature_num, feature_columns, dnn_dropout, emb_size, hidden_units, field_num, implicit_vector_dim, device
        model = DeepFFMS(22, 40, feature_columns, dnn_dropout, emb_size, hidden_units, field_num,implicit_vector_dim, device).to(device)

        # for name, para in model.named_parameters():
        #     if name == "ffm.a":
        #         para.requires_grad = False
        #     print(name)
        #     print(para)

        log.info(model)
        loss_func = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(params=filter(lambda para: para.requires_grad, model.parameters()), lr=0.001,
                                     weight_decay=0.0005)
        metric_func = self.auc
        metric_name = 'auc'
        # 控制步长
        epochs = 50
        # 每几步打印下日志
        log_step_freq = 50
        dfhistory = pd.DataFrame(columns=['epoch', 'loss', metric_name, 'val_loss', 'val_' + metric_name])

        log.info('start_training.........')
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log.info('========' * 8 + '%s' % nowtime)

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

                try:
                    loss = loss_func(predictions.double().cpu(), labels.double().cpu())
                except:
                    log.info(predictions.min())
                    log.info(labels)

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
                    log.info(("[step=%d] loss: %.3f, " + metric_name + ": %.3f") % (
                    step, loss_sum / step, metric_sum / step))

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
            log.info((
                                 "\nEPOCH=%d, loss=%.3f, " + metric_name + " = %.3f, val_loss=%.3f, " + "val_" + metric_name + " = %.3f") % info)
            nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log.info('\n' + '==========' * 8 + '%s' % nowtime)

        log.info('Finished Training')
        log.info(dfhistory)
        log.save()
        # 观察损失和准确率的变化
        self.plot_metric(dfhistory, "loss", f"../log/{log_folder_name}/loss.png")
        self.plot_metric(dfhistory, "auc", f"../log/{log_folder_name}/auc.png")
        with open(f"../log/{log_folder_name}/metric.txt", mode='w', encoding='utf-8') as f:
            f.write(str(dfhistory))

        # 预测
        # y_pred_probs = model(torch.tensor(test, device=device).float())
        # y_pred = torch.where(y_pred_probs > 0.5, torch.ones_like(y_pred_probs), torch.zeros_like(y_pred_probs))
        # print(y_pred.data)

    def plot_metric(self, dfhistory, metric, file_path):
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
        plt.savefig(file_path)


    # 模型的相关设置
    def auc(self, y_pred, y_true):
        pred = y_pred.data
        y = y_true.data
        return roc_auc_score(y, pred)

    def read_data(self):
        # 读入训练集，验证集和测试集
        train = pd.read_csv(self.dataset_file_path + '/train_10w.csv')
        val = pd.read_csv(self.dataset_file_path + '/val_10w.csv')

        train.columns = [index for index in range(0, self.all_feature_num)]
        val.columns = [index for index in range(0, self.all_feature_num)]

        # 统计每种离散特征有多少枚举值
        all = pd.concat([train, val])
        # print(all)
        feature_columns = [all[index].max() + 1 for index in range(1, self.feature_num)]
        print("feature_columns:" + str(feature_columns))
        # print(train)
        train_x, train_y = train.drop(columns=0).values, train[0].values
        # print(train_y)
        val_x, val_y = val.drop(columns=0).values, val[0].values

        return train_x, train_y, val_x, val_y, feature_columns

    def set_log(self, log_folder_name, do=True):
        if do is False:
            return Log(f"../log/{log_folder_name}" + "/terminal.log")
        # 建立文件夹
        os.mkdir('../log/' + log_folder_name)
        # 创建三个文件
        # 第一个文件存储所有参数值
        file = open(f"../log/{log_folder_name}/parameter.yaml", 'w')
        file.close()
        # 第二个文件存储所有控制台输出
        print_log = open(f"../log/{log_folder_name}/terminal.log", "w")
        print_log.close()
        # 第三个存储指标值
        file = open(f"../log/{log_folder_name}/metric.txt", 'w')
        file.close()

        return Log(f"../log/{log_folder_name}" + "/terminal.log")


def run():
    train = Train("../project/dataset/avazu", 23, 63)

    train.train()


if __name__ == '__main__':
    run()