import numpy as np
import pandas as pd
import sys
sys.path.append('../')
sys.path.append('../code')
from util import *
from NeuralNet import *

def configureFile():
        return{
        "trainFile":"data/train.csv",
        "outputFile":"check/18308133_liuxianbin_lr",
        "trainMode":1,
        "miniBatch": 100,
        "k":7,
        "iter":100,
        "lrate":0.01
    }

def main():
    
    dataset = pd.read_csv("check/train.csv", header=None).values
    dataset[:,-1] = dataset[:,-1].astype('int')
    loader = DataLoader(dataset)
    for trainset, trlabel, valset, valabel in loader.KfolderData(7):
        dims = len(trainset[0])
        lrMod = NerualNet(dims)
        lrMod.train(trainset, labels=trlabel, lrate=lrate, iters=config["iter"])
        lrMod.eval()
        pred, ac = lrMod.predict(valset, valabel)
        result = np.append(valabel, pred, axis=1)
        print("ac of lrate:%f is %f" %(lrate, ac))
