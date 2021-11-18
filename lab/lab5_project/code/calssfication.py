from datetime import date
from nltk.corpus.util import TRY_ZIPFILE_FIRST
import numpy as np
from numpy.lib.twodim_base import tri
import pandas as pd
from typing import Iterable, List, Union, ValuesView
import matplotlib.pyplot as plt
import sys
import copy
import re
from pandas.core.algorithms import mode

from pandas.core.frame import DataFrame
from torch.nn import parameter
from torch.optim import optimizer
sys.path.append('../')
sys.path.append('../code')
from util import *
from NeuralNet import *

def configFile()->dict[str, str]:
    return{
        "trainFile":"data/classification/train_vector002.csv",
        "testFile":"data/classification/test_vector002.csv",
        "predTestFile":"result/cls_020.csv",
        "paramFile":"param/classification/020",
        #"paramFile":"torchtest",
        "k":20,
        "lrate":0.01,
        "epochs":20,
        "batchSize":128,
        "shuffle":False,
        "load":False,   
        "HiddenDim":[64],
        'activation' : ['sigmod','sigmod'],
        "log":False,
        'bn':True,
        'lossfunction':"MSELoss",
        'optim':"adam"
    }

def main():


    config = configFile()
    lrate, epochs, batchSize = config["lrate"], config["epochs"], config["batchSize"]

    '''-----确定好超参数----'''
    config["test"]=True
    config["lastACT"] = "sigmod"

    # 读入vetcor文件，此时带有label以及index
    trainset = pd.read_csv(config["trainFile"], header=None).values[:,1:]    # 去除textid
    loader = DataLoader(trainset)

    testSet = pd.read_csv(config["testFile"], header=None).values

    # Kfloder分数据
    for traindata, trlabel, valdata, valabel in loader.KfolderData(config["k"], shuffle=False, test=config["test"]):
        dims = len(traindata[0])
        # 建立NeuralNet模型
        if config['load']:
            model = NerualNet(paramsFile=config["paramFile"])
        else:
            model = NerualNet(featureDim=dims, outputDim=1, HiddenDims=copy.deepcopy(config['HiddenDim']))
            model.saveParam(config["paramFile"])
        
        # model.addActivation("Sigmod")
        # model.addActivation("ReLU")
        # model.addActivation("sigmod")
        model.addActivation(config["activation"])

        # eval with torch, should be delete
        #model = NerualNet(paramsFile="torchtest.npz", lastACT=config["lastACT"])
        
        # for i in range(model.layers):
        #     model.W[i] = model.W[i].T
        #     model.b[i] = model.b[i].reshape((1,-1))

        # 训练
        model.groupReLU = (1, 0.1)
        lratestragety = LrateStrategy(model.W, model.b)
        optim = None
        if config['optim'].lower() == 'adam':
            optim = lratestragety.Adam()
        elif config['optim'].lower() == 'adagrad':
            optim = lratestragety.Adagrad()
        elif config['optim'].lower() == 'rmsprop':
            optim = lratestragety.RMSprop()
        Loss, ac_t, ac_v = model.train(traindata, trlabel, valdata, valabel, lrate=lrate, epochs=epochs, batchSize=batchSize, lossfunction=config["lossfunction"], optim=optim, lamda=0.0, bn=config["bn"])
        plotTraining(Loss, ac_t, ac_v)

        # 预测
        pred,ac = model.predict(traindata, trlabel, ac=True, bn=config["bn"])
        print(f"the ac of config in tr: lrate:{lrate}, epochs:{epochs}, batchSize:{batchSize} is {ac}")
        config['accuaracy in trainset'] = ac

        pred,ac = model.predict(valdata, valabel, ac=True, bn=config["bn"])
        print(f"the ac of config in val: lrate:{lrate}, epochs:{epochs}, batchSize:{batchSize} is {ac}")
        config['accuaracy in valset'] = ac

        if ac > 0.855: 
            config['log']=True

    print("\n\n 现在开始测试：测试集...... \n\n")
    pred = model.predict(testSet[:,1:], bn=config["bn"])
    pred = (pred > 0.5).astype("int")
    testdata = np.append(testSet[:,0].reshape((-1,1)).astype("int"), pred, axis=1)


    if config["log"]:
        DataFrame(testdata, columns=["id","is_humor"]).to_csv(config["predTestFile"], index=None)
    
    if config["log"]:
        logtext = "本次运行的配置文件如下："
        log(logtext, config, "log/classification/log.txt")

if __name__ == "__main__":
    main()