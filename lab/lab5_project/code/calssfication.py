import numpy as np
from numpy.lib.twodim_base import tri
import pandas as pd
from typing import Iterable, List, Union
import matplotlib.pyplot as plt
import sys
import copy
import re

from pandas.core.frame import DataFrame
sys.path.append('../')
sys.path.append('../code')
from util import *
from NeuralNet import *

def configFile()->dict[str, str]:
    return{
        "trainFile":"data/classification/train_vector002.csv",
        "testFile":"data/classification/test_vector002.csv",
        "predTestFile":"result/cls_003.csv",
        "paramFile":"param/classification/003",
        "k":10,
        "lrate":0.5,
        "epochs":400,
        "batchSize":128,
        "shuffle":True,
        "Hidden":[128],
        "log":False
    }

def main():
    
    config = configFile()
    lrate, epochs, batchSize = config["lrate"], config["epochs"], config["batchSize"]

    # 读入vetcor文件，此时带有label以及index
    trainset = pd.read_csv(config["trainFile"], header=None).values[:,1:]    # 去除textid
    loader = DataLoader(trainset)
    
    testSet = pd.read_csv(config["testFile"], header=None).values

    '''-----确定好超参数----'''
    config["test"]=True
    config["lastACT"] = "sigmod"
    config["lossfunction"] = "MLELoss"
    # Kfloder分数据
    for traindata, trlabel, valdata, valabel in loader.KfolderData(config["k"], shuffle=False, test=config["test"]):
        dims = len(traindata[0])
        # 建立NeuralNet模型
        model = NerualNet(featureDim=dims, outputDim=1, HiddenDims=copy.deepcopy(config["Hidden"]), lastACT=config["lastACT"])
        # 训练
        Loss, ac_t, ac_v = model.train(traindata, trlabel, valdata, valabel, lrate=lrate, epochs=epochs, batchSize=batchSize, lossfunction=config["lossfunction"])
        plotTraining(Loss, ac_t, ac_v)

        # 预测
        pred,ac = model.predict(valdata, valabel, ac=True)
        print(f"the ac of config: lrate:{lrate}, epochs:{epochs}, batchSize:{batchSize} is {ac}")


    model.saveParam(config["paramFile"])
    print("\n\n 现在开始测试：测试集...... \n\n")
    pred = model.predict(testSet[:,1:])
    pred = (pred > 0.5).astype("int")
    testdata = np.append(testSet[:,0].reshape((-1,1)).astype("int"), pred, axis=1)
    DataFrame(testdata, columns=["id","is_humor"]).to_csv(config["predTestFile"], index=None)
    
    if config["log"]:
        logtext = "本次运行的配置文件如下："
        log(logtext, config, "log/classification/log.txt")

if __name__ == "__main__":
    main()