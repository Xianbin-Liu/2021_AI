import numpy as np
import pandas as pd
from typing import Iterable, List, Union
import matplotlib.pyplot as plt
import sys
import re

from pandas.core.frame import DataFrame
sys.path.append('../')
sys.path.append('../code')
from util import *
from NeuralNet import *

def configFile()->dict[str, str]:
    return{
        "trainFile":"data/regression/train_vector.csv",
        "testFile":"data/regression/test_vector.csv",
        "predTestFile":"result/reg_001.csv",
        "k":10,
        "lrate":0.01,
        "epochs":200,
        "batchSize":256
    }

def main():
    
    config = configFile()
    lrate, epochs, batchSize = config["lrate"], config["epochs"], config["batchSize"]

    # 读入vetcor文件，此时带有label以及index
    trainset = pd.read_csv(config["trainFile"], header=None).values[:,1:]    # 去除textid
    loader = DataLoader(trainset)

    testSet = pd.read_csv(config["testFile"], header=None).values

    '''-----确定好超参数----'''

    # Kfloder分数据
    for traindata, trlabel, valdata, valabel in loader.KfolderData(config["k"], shuffle=True, test=True):
        dims = len(traindata[0])
        # 建立NeuralNet模型
        model = NerualNet(featureDim=dims, outputDim=1, HiddenDims=[64], lastACT="linear")

        # 训练
        Loss, ac_t, ac_v = model.train(traindata, trlabel, valdata, valabel, lrate=lrate, epochs=epochs, batchSize=batchSize, lossfunction="RMSELoss")
        plotTraining(Loss, ac_t, ac_v)

        # 预测  
        pred,loss = model.predict(valdata, valabel)
        print(f"the loss of config: lrate:{lrate}, epochs:{epochs}, batchSize:{batchSize} is {loss}")

        print("\n\n 现在开始测试：测试集...... \n\n")
        pred = model.predict(testSet[:,1:])
        testdata = np.append(testSet[:,0].reshape((-1,1)), pred, axis=1)
        DataFrame(testdata, columns=["id","humor_rating"]).to_csv(config["predTestFile"], index=None)

if __name__ == "__main__":
    main()