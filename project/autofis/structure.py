import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split



import matplotlib.pyplot as plt
def plot_metric(dfhistory):
    metric = 'AUC'
    train_metrics = dfhistory[metric]
    epochs = list(dfhistory['Shape'])
    plt.plot(epochs, train_metrics, 'bo--')
    # plt.title('Training and validation ' + metric)
    plt.xlabel("Shape")
    plt.ylabel(metric)
    # plt.legend(["train_" + metric, 'val_' + metric])
    plt.show()
ss = """constant 0.778237
diamond 0.778392
increase 0.778166
decrease 0.778635"""

import pandas as pd
import math
dfhistory = pd.DataFrame(columns=['Shape', 'AUC'])
for i, index in enumerate(ss.split("\n")):
    val = index.split(' ')
    dfhistory.loc[i] = (val[0], float(val[1]))


print(dfhistory)

plot_metric(dfhistory)


