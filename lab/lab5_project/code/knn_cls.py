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
from knn import *

config = {
    "trainFile":"data/classification/train_vector002.csv",
    "testFile":"data/classification/test_vector002.csv",
    "predTestFile":"result/cls_004_knn.csv",
    "k":10,
    "lrate":0.01,
    "epochs":10,
    "batchSize":128,
    "shuffle":True,
    "load":False,
    "HiddenDim":[64],
    "log":True
}

trainset = pd.read_csv(config["trainFile"], header=None).values[:,1:]    # 去除textid
loader = DataLoader(trainset)

testSet = pd.read_csv(config["testFile"], header=None).values

for trdata, trlabel, valset, valable in loader.KfolderData(config['k'], shuffle=config['shuffle'], test=False):
    model = KNN(k=7)
    model.train(trdata, trlabel)
    
    yp = model.predict(valset, k=7, lossType=3)
    ac = (yp == valable).mean()
    print(ac)