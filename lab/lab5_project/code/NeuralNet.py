from os import linesep
import numpy as np
from numpy.core.fromnumeric import trace
from numpy.lib.function_base import blackman
from numpy.lib.twodim_base import triu_indices_from
from numpy.random.mtrand import set_state
import pandas as pd
from typing import Iterable
import sys
sys.path.append('../')
sys.path.append('../code')
from util import *

def configFile():
    return{
        "trainFile":"data/train.csv",
        "outputFile":"check/18308133_liuxianbin_lr",
        "trainMode":1,
        "miniBatch": 100,
        "k":7,
        "iter":100,
        "lrate":0.01
    }

# use to debug with one layer : y = Wx+b
class NerualNet:
    def __init__(self, dims) -> None:
        self.dims = dims + 1    # merge with b
        self.W = np.zeros((self.dims, 1))
        #self.W = np.random.random((self.dims, 1)) # of shape (1, dims)
        self.trainMode = True
    
    def eval(self):
        self.trainMode = False

    def trainning(self):
        self.trainMode = True

    def Loss(self, X, Y=None, epsilonmax=0.95, epsilonmin=0.05):
        # Y1 = self.W.dot(X)
        Y1,X1 = self.Fullnet(X)
        Y2,X2 = self.sigmod(Y1)
        Y2 = np.minimum(np.maximum(Y2, epsilonmin), epsilonmax)
        loss, X3 = self.MLELoss(Y2, Y)   # X3 = Y2
        if not self.trainMode or Y is None:
            return Y2, loss
        # backward
        dy1 = self.backwardSigmodMLELoss(Y2, Y)
        dW, dy2 = self.backwardFullnet(self.W, X1, dy1)
        dW /= len(X)
        return dW, loss

    def train(self, X, labels=None, lrate=0.01, iters=10, batchSize=0, epsilonmax=0.95, epsilonmin=0.05):
        # ensure X NOT append with label
        X = np.append(X, np.ones((len(X),1)), axis=1)
        if batchSize == 0: batchSize = len(X)
        Loss = []
        for iter in range(iters):
            if labels is None: labels =np.ones((len(X),1))
            for miniX, minY in DataLoader(np.append(X,labels, axis=1)).Batch(batchSize):
                dW, loss = self.Loss(miniX, minY)
                self.W -= lrate*(dW)

    # return Y, X
    def Fullnet(self, X):
        return X[:,:self.dims].dot(self.W), X


    def backwardFullnet(self, W, X, dy_upper):
        # return dW AND dy
        dW = X.T.dot(dy_upper)
        dy = W
        return dW, dy

    def sigmod(self,X):
        return 1/(1+np.exp(-X)), X

    def backwardSigmodFullnet(self, dy_upper, Y, label):
        pass

    def backwardSigmodMLELoss(self, Y, label):
        return -(label-Y)


    def ReLU(self, X):
        return (X*(X>0)).astype('np.float64'), X

    # no Param
    def backwardReLU(self, X, dy_upper):
        # dy is upper derivate
        # return d(ReLu_y)/d(ReLU_x)
        return (X > 0).dot(dy_upper)

    def backwardSigmod(self, Y, dy_upper):
        # dy = -label*(1/Y) + (1-label)*(1/(1-Y))
        # so, backwardLogi return -(label*(1-Y) - (1-label)*Y) = -(label-Y)
        return Y*(1-Y)*(dy_upper)
    
    # end Point
    def MLELoss(self, X, Y):
        return -(Y.T.dot(np.log(X))+(1-Y).T.dot(np.log(1-X))) / len(X), X

    # end Point
    def backwardMLELoss(self, Y, label):
        return -label*(1/Y) + (1-label)*(1/(1-Y))
    
    def RMSELoss(self, X, Y):
        return np.sqrt(np.sum(np.square(X)+np.square))

    def backwardRMSE(self):
        pass

    def predictSingle(self, data):
        return self.sigmod(data[:self.dims].dot(self.W))

    def predict(self, datas:Iterable, labels=None):
        # fit
        self.eval()
        res, loss = self.Loss(np.append(datas, np.ones((len(datas),1)), axis=1), labels)
        if labels is not None:
            res = (res > 0.5).astype('int').reshape((-1, 1))
            return res, (res==labels).mean()
        else:
            return res


def main2():
    config = configFile()
    lrate=config['lrate']
    # trainset = pd.read_csv("check/train_sample.csv", header=None).values
    # valset =  pd.read_csv("check/test.csv", header=None).values
    dataset = pd.read_csv("check/train.csv", header=None).values
    dataset[:,-1] = dataset[:,-1].astype('int')
    # np.random.shuffle(dataset)
    # trainset = dataset[:-1000,:]
    # valset = dataset[-1000:,:]
    # trlabel = trainset[:,-1].reshape(-1,1)
    # trainset = trainset[:,:-1]
    # valabel = valset[:,-1].reshape(-1,1)
    # valset = valset[:,:-1]
    # dims = len(trainset[0])
    loader = DataLoader(dataset)
    for trainset, trlabel, valset, valabel in loader.KfolderData(7):
        dims = len(trainset[0])
        lrMod = NerualNet(dims)
        lrMod.train(trainset, labels=trlabel, lrate=lrate, iters=config["iter"])
        lrMod.eval()
        pred, ac = lrMod.predict(valset, valabel)
        result = np.append(valabel, pred, axis=1)
        print("ac of lrate:%f is %f" %(lrate, ac))

if __name__ == "__main__":
    main2()