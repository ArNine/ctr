import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split



import matplotlib.pyplot as plt
def plot_metric(dfhistory):
    metric = 'AUC'
    train_metrics = dfhistory[metric]
    train_metrics2 = dfhistory['ff']
    epochs = list(dfhistory['Dropout'])
    plt.plot(epochs, train_metrics, 'bo--')
    # plt.plot(epochs, train_metrics2, 'go--')
    # plt.title('AUC curves with different Dropout values')
    plt.xlabel("Dropout")
    plt.ylabel(metric)
    # plt.legend(['Inner Product', 'Outer Product'])
    plt.show()
ss = """0.1 0.776933 0.777033
0.2 0.777938 0.778038
0.3 0.776641 0.776041
0.4 0.777601 0.777401
0.5 0.777282 0.776882
0.6 0.778635 0.778035
0.7 0.777158 0.777058
0.8 0.776395 0.777395
0.9 0.772819 0.773119"""

import pandas as pd
import math
dfhistory = pd.DataFrame(columns=['Dropout', 'AUC', 'ff'])
for i, index in enumerate(ss.split("\n")):
    val = index.split(' ')
    dfhistory.loc[i] = (float(val[0]), float(val[1]), float(val[2]))


print(dfhistory)

plot_metric(dfhistory)


