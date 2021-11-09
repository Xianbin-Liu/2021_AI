import numpy as np
import pandas as pd
from typing import Iterable, List, Union
import matplotlib.pyplot as plt
import sys
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
        "trainFile":"data/regression/train_vector002.csv",
        "testFile":"data/regression/test_vector002.csv",
        "predTestFile":"result/reg_001.csv",
        "paramFile":"param/regression/001",
        "k":100,
        "lrate":0.01,
        "epochs":88,
        "batchSize":100,
        "HiddenDim":[64],
        "shuffle":True,
        "load":False,
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
    config["lastACT"]='linear'
    config["lossfunction"] = 'MSELoss'
    # Kfloder分数据
    for traindata, trlabel, valdata, valabel in loader.KfolderData(config["k"], shuffle=config["shuffle"], test=config["test"]):
        dims = len(traindata[0])
        # 建立NeuralNet模型
        if not config['load']:
            model = NerualNet(featureDim=dims, outputDim=1, HiddenDims=config['HiddenDim'], lastACT=config["lastACT"])
        else:
            model = NerualNet(paramsFile=config["paramFile"])
        # 训练
        lrs = LrateStrategy(model.W, model.b)
        adam = lrs.Adam()
        Loss, ac_t, ac_v = model.train(traindata, trlabel, valdata, valabel, lrate=lrate, epochs=epochs, batchSize=batchSize, lossfunction=config["lossfunction"], optim=adam)
        plotTraining(Loss, ac_t, ac_v)

        # 预测  
        pred,loss = model.predict(valdata, valabel)
        print(f"the loss of config: lrate:{lrate}, epochs:{epochs}, batchSize:{batchSize} is {loss}")
        

    if config['log']:
        print("\n\n 现在开始测试：测试集...... \n\n")
        model.saveParam(config["paramFile"])
        pred = model.predict(testSet[:,1:])
        df = DataFrame(columns=["id","humor_rating"], index=None)
        df['id'] = testSet[:,0].astype('int')
        df['humor_rating'] = pred
        df.to_csv(config['predTestFile'], index=None)
        log("这次的配置如下：",config, file="log/regression/log.txt")

if __name__ == "__main__":
    main()