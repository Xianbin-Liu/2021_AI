from nltk.corpus.util import TRY_ZIPFILE_FIRST
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
        "predTestFile":"result/reg_008.csv",
        "paramFile":"param/regression/008",
        "k":20,
        "lrate":0.01,
        "epochs":30,
        "batchSize":256,
        "HiddenDim":[64],
        'activation' : ['sigmod','linear'],
        "shuffle":False,
        "load":False,
        "log":False,
        "relu":(1,0),
        "bn":False
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
            model.saveParam(config["paramFile"])
        else:
            model = NerualNet(paramsFile=config["paramFile"])
        
        model.addActivation(config["activation"])
        model.groupReLU = config['relu']
        # 训练
        lrs = LrateStrategy(model.W, model.b)
        adam = lrs.Adam()
        Loss, ac_t, ac_v = model.train(traindata, trlabel, valdata, valabel, lrate=lrate, epochs=epochs, batchSize=batchSize, lossfunction=config["lossfunction"], optim=adam, bn=config["bn"])
        plotTraining(Loss, ac_t, ac_v)

        # 预测  
        pred,loss = model.predict(valdata, valabel, bn=config['bn'])
        print(f"the loss of config: lrate:{lrate}, epochs:{epochs}, batchSize:{batchSize} is {loss}")
        

    if config['log']:
        print("\n\n 现在开始测试：测试集...... \n\n")
        pred = model.predict(testSet[:,1:], bn=config['bn'])
        df = DataFrame(columns=["id","humor_rating"], index=None)
        df['id'] = testSet[:,0].astype('int')
        df['humor_rating'] = pred
        df.to_csv(config['predTestFile'], index=None)
        log("这次的配置如下：",config, file="log/regression/log.txt")

if __name__ == "__main__":
    main()