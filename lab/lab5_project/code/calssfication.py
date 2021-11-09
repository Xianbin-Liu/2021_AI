from datetime import date
import numpy as np
from numpy.lib.twodim_base import tri
import pandas as pd
from typing import Iterable, List, Union
import matplotlib.pyplot as plt
import sys
import copy
import re
from pandas.core.algorithms import mode

from pandas.core.frame import DataFrame
from torch.nn import parameter
sys.path.append('../')
sys.path.append('../code')
from util import *
from NeuralNet import *

def configFile()->dict[str, str]:
    return{
        "trainFile":"data/classification/train_vector002.csv",
        "testFile":"data/classification/test_vector002.csv",
        "predTestFile":"result/cls_004.csv",
        "paramFile":"param/classification/004",
        "k":10,
        "lrate":0.01,
        "epochs":10,
        "batchSize":128,
        "shuffle":True,
        "load":False,
        "HiddenDim":[64],
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
    config["optimize"] = "adam"
    # Kfloder分数据
    for traindata, trlabel, valdata, valabel in loader.KfolderData(config["k"], shuffle=False, test=config["test"]):
        dims = len(traindata[0])
        # 建立NeuralNet模型
        if config['load']:
            model = NerualNet(paramsFile=config["paramFile"])
        else:
            model = NerualNet(featureDim=100, outputDim=1, HiddenDims=config['HiddenDim'], lastACT=config["lastACT"])
        
        # eval with torch, should be delete
        #model = NerualNet(paramsFile="torchtest.npz", lastACT=config["lastACT"])
        
        # for i in range(model.layers):
        #     model.W[i] = model.W[i].T
        #     model.b[i] = model.b[i].reshape((1,-1))

        # 训练
        lratestragety = LrateStrategy(model.W, model.b)
        optim = lratestragety.Adam()
        Loss, ac_t, ac_v = model.train(traindata, trlabel, valdata, valabel, lrate=lrate, epochs=epochs, batchSize=batchSize, lossfunction=config["lossfunction"], optim=optim, lamda=0.0)
        plotTraining(Loss, ac_t, ac_v)

        # 预测
        pred,ac = model.predict(valdata, valabel, ac=True)
        print(f"the ac of config: lrate:{lrate}, epochs:{epochs}, batchSize:{batchSize} is {ac}")

    if config["log"]:
        model.saveParam(config["paramFile"])
    print("\n\n 现在开始测试：测试集...... \n\n")
    pred = model.predict(testSet[:,1:])
    pred = (pred > 0.5).astype("int")
    testdata = np.append(testSet[:,0].reshape((-1,1)).astype("int"), pred, axis=1)
    if config["log"]:
        DataFrame(testdata, columns=["id","is_humor"]).to_csv(config["predTestFile"], index=None)
    
    if config["log"]:
        logtext = "本次运行的配置文件如下："
        log(logtext, config, "log/classification/log.txt")

if __name__ == "__main__":
    main()