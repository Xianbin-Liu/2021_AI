from os import linesep
import numpy as np
from numpy.core.fromnumeric import trace
from numpy.core.numeric import ones
from numpy.lib.function_base import blackman
from numpy.lib.twodim_base import tril, triu_indices_from
from numpy.random.mtrand import set_state
import pandas as pd
from typing import Iterable, List, Union
import matplotlib.pyplot as plt
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
        "lrate":0.1
    }

# use to debug with one layer : y = Wx+b
class NerualNet:
    def __init__(self, featureDim, outputDim=1, HiddenDims=[], actiF=0, weight_scale=1e-2) -> None:
        self.dims = featureDim   # merge with b
        self.W = []
        self.b = []
        HiddenDims.insert(0, featureDim)
        HiddenDims.append(outputDim)
        for i in range(len(HiddenDims)-1):
            self.W.append(weight_scale*np.random.randn(HiddenDims[i], HiddenDims[i+1]))  #dont merge
            self.b.append(weight_scale*np.random.randn(1, HiddenDims[i+1]))
        #self.W = np.random.random((self.dims, 1)) # of shape (1, dims)
        self.layers = len(self.W)
        self.trainMode = True
    
    def eval(self):
        self.trainMode = False

    def trainning(self):
        self.trainMode = True

    def loadParam(self, filePath):
        

    def Loss(self, X, Labels=None, epsilonmax=0.95, epsilonmin=0.05):
        # Y1 = self.W.dot(X)
        dW_all, db_all, X_fullnet, Y_Fullnet, Y_act = [],[],[],[],[]
        Y = None  # predict

        # forward
        for i in range(self.layers):
            Y,X = self.Fullnet(X,self.W[i],self.b[i])
            X_fullnet.append(X)
            Y_Fullnet.append(Y)
            Y,X2 = self.sigmod(Y)
            Y = np.minimum(np.maximum(Y, epsilonmin), epsilonmax)
            Y_act.append(Y)    # output of each layer
            X = Y   #  Next layer input

        if not self.trainMode or Labels is None:
            return Y

        loss = self.MLELoss(Y, Labels)

        # backward
        dy = self.backwardSigmodMLELoss(Y, Labels)
        dW, db, dy = self.backwardFullnet(self.W[-1], X_fullnet[-1], dy)
        dW_all.insert(0, dW)
        db_all.insert(0, db)
        for i in range(self.layers-2, -1, -1):
            dy = self.backwardSigmod(Y_act[i], dy)
            dW, db, dy = self.backwardFullnet(self.W[i], X_fullnet[i], dy)
            dW_all.insert(0, dW)
            db_all.insert(0, db)

        return dW_all, db_all, loss

    def train(self, X, labels=None, valSet=None, valabel=None, lrate=0.01, iters=10, batchSize=0, epsilonmax=0.95, epsilonmin=0.05):
        # ensure X NOT append with label
        if batchSize == 0: batchSize = len(X)
        Loss = []
        ac_t = []
        ac_v = []
        cnt = 0
        for iter in range(iters):
            if labels is None: labels =np.ones((len(X),1))
            for miniX, minY in DataLoader(np.append(X,labels, axis=1)).Batch(batchSize,shuffle=True):
                cnt += 1
                dW, db, loss = self.Loss(miniX, minY)
                for i in range(self.layers):
                    self.W[i] -= lrate*dW[i]
                    self.b[i] -= lrate*db[i]
                if cnt % 10 == 0:
                    Loss.append(loss)
                    ac = ((self.Loss(miniX)>0.5).astype('int').reshape(-1,1)==minY).mean()
                    ac_t.append(ac)
                    if valSet is not None:
                        ac = ((self.Loss(valSet)>0.5).astype('int').reshape(-1,1)==valabel).mean()
                        ac_v.append(ac)
                    
        return Loss, ac_t, ac_v

    # return Y, X
    def Fullnet(self, X, W, b):
        return X.dot(W)+b, X


    def backwardFullnet(self, W, X, dy_upper):
        # return dW AND dy
        dW = X.T.dot(dy_upper) / len(X)
        db = np.ones((1,len(X))).dot(dy_upper) / len(X)
        dy = dy_upper.dot(W.T)
        return dW, db, dy

    def sigmod(self,X):
        return 1/(1+np.exp(-X)), X

    def backwardSigmodFullnet(self, dy_upper, Y):
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
        return -(Y.T.dot(np.log(X))+(1-Y).T.dot(np.log(1-X))) / len(X)

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
        res = self.Loss(datas, labels)
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
    for trainset, trlabel, valset, valabel in loader.KfolderData(7, shuffle=True):
        dims = len(trainset[0])
        lrMod = NerualNet(dims, outputDim=1, HiddenDims=[])
        Loss, ac_t, ac_v = lrMod.train(trainset, labels=trlabel, lrate=lrate, iters=config["iter"], batchSize=1000)
        x1 = np.arange(0, len(Loss), 1)
        plt.subplot(1,3,1)
        plt.scatter(x1, Loss)
        plt.title("Loss")
        
        if ac_v != []:
            plt.subplot(1,3,2)
            plt.scatter(x1, ac_v)
            plt.title("ac_v")

        plt.subplot(1,3,3)
        plt.scatter(x1, ac_t)
        plt.title("ac_t")

        plt.show()

        lrMod.eval()
        pred, ac = lrMod.predict(valset, valabel)
        result = np.append(valabel, pred, axis=1)
        print("ac of lrate:%f is %f" %(lrate, ac))

if __name__ == "__main__":
    main2()