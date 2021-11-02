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
import re
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
        "iter":300,
        "lrate":0.1
    }

# use to debug with one layer : y = Wx+b
class NerualNet:
    def __init__(self, paramsFile=None, featureDim=None, outputDim=1, HiddenDims=[], actiF=0, weight_scale=1e-2) -> None:
        self.layers = self.dims = 0
        self.W = []
        self.b = []
        if paramsFile != None:
           self.loadParam(paramsFile)
           self.dims = len(self.W[0])
        else:
            self.dims = featureDim   # merge with b
            HiddenDims.insert(0, featureDim)
            HiddenDims.append(outputDim)
            for i in range(len(HiddenDims)-1):
                self.W.append(weight_scale*np.random.randn(HiddenDims[i], HiddenDims[i+1]))  #dont merge
                self.b.append(weight_scale*np.random.randn(1, HiddenDims[i+1]))
            #self.W = np.random.random((self.dims, 1)) # of shape (1, dims)
        self.layers = len(self.W)
        self.trainMode = True
    
    @property
    def param(self):
        param = {}
        for i in range(self.layers):
            param["W"+str(i+1)] = self.W[i]
            param["b"+str(i+1)] = self.b[i]
        return param

    def eval(self):
        self.trainMode = False

    def trainning(self):
        self.trainMode = True

    def loadParam(self, file):
        R = ".*\.npz"
        rule = re.compile(R)
        if rule.match(file) is None:
            file += ".npz"
        param = np.load(file)
        self.layers = int(len(param.files)/2)
        if self.W : del self.W
        if self.b : del self.b
        self.W = []
        self.b = []
        for i in range(self.layers):
            self.W.append(param["W"+str(i+1)])
            self.b.append(param["b"+str(i+1)])      

    def saveParam(self, file):
        np.savez(file, **self.param)
        

    def Loss(self, X, Labels=None, epsilonmax=0.95, epsilonmin=0.05, lossfunction="RMSELoss"):
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

        # perform Loss
        if lossfunction == "MLELoss":
            loss = self.MLELoss(Y, Labels)

            # first lay backard with Loss
            dy = self.backwardSigmodMLELoss(Y, Labels)
            dW, db, dy = self.backwardFullnet(self.W[-1], X_fullnet[-1], dy)
        elif lossfunction == "RMSELoss":
            loss = self.RMSELoss(Y, Labels)
            dy = self.backwardRMSE(loss, Y, Labels)
            dy = self.backwardSigmod(Y, dy)
            dW, db, dy = self.backwardFullnet(self.W[-1], X_fullnet[-1], dy)

        dW_all.insert(0, dW)
        db_all.insert(0, db)
        for i in range(self.layers-2, -1, -1):
            dy = self.backwardSigmod(Y_act[i], dy)
            dW, db, dy = self.backwardFullnet(self.W[i], X_fullnet[i], dy)
            dW_all.insert(0, dW)
            db_all.insert(0, db)

        return dW_all, db_all, loss

    def train(self, X, labels=None, valSet=None, valabel=None, lrate=0.01, iters=10, batchSize=0, epsilonmax=0.95, epsilonmin=0.05, lossfunction="RMSELoss"):
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
                dW, db, loss = self.Loss(miniX, minY, lossfunction=lossfunction)
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
    
    def RMSELoss(self, Yp, Y):
        return np.sqrt(Y.T.dot(Y)+Yp.T.dot(Yp)-2*Y.T.dot(Yp))/len(Y)

    def backwardRMSE(self, Loss, Yp, Y):
        return (Yp-Y)/Loss
    
    def backwardSigmodRMSE(self):
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
    dataset = pd.read_csv("check/train.csv", header=None).values
    dataset[:,-1] = dataset[:,-1].astype('int')
    loader = DataLoader(dataset)
    for trainset, trlabel, valset, valabel in loader.KfolderData(7, shuffle=False, test=True):
        dims = len(trainset[0])
        lrMod = NerualNet(featureDim=dims, outputDim=1, HiddenDims=[50, 20])
        Loss, ac_t, ac_v = lrMod.train(trainset, labels=trlabel, lrate=lrate, iters=config["iter"], batchSize=1000, lossfunction="RMSELoss")
        #lrMod.loadParam("test")
        # x1 = np.arange(0, len(Loss), 1)
        # plt.subplot(1,3,1)
        # plt.scatter(x1, Loss)
        # plt.title("Loss")
        
        # if ac_v != []:
        #     plt.subplot(1,3,2)
        #     plt.scatter(x1, ac_v)
        #     plt.title("ac_v")

        # plt.subplot(1,3,3)
        # plt.scatter(x1, ac_t)
        # plt.title("ac_t")

        # plt.show()
        lrMod.eval()
        pred, ac = lrMod.predict(valset, valabel)
        result = np.append(valabel, pred, axis=1)
        print("ac of lrate:%f is %f" %(lrate, ac))

if __name__ == "__main__":
    main2()
    # model = NerualNet(3, 1, [3])
    # model.W[1] = np.zeros(shape=model.W[1].shape)
    # print(model.param)
    # model.saveParam("test")
    # model2 = NerualNet(3, 1, [3])
    # model2.loadParam("test")
    # print(model2.param)
    