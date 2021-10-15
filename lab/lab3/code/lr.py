from functools import cache
from os import RTLD_GLOBAL, name
import numpy as np
from numpy.lib.function_base import average
from numpy.random.mtrand import random
import pandas as pd
import sys
from typing import Iterable, Sized, Union

from pandas.core.algorithms import SelectNFrame, value_counts

sys.path.append('../')
sys.path.append('../code')
from util import DataLoader

def configFile():
    return{
        "trainFile":"data/train.csv",
        "outputFile":"check/18308133_liuxianbin_lr",
        "trainMode":1,
        "miniBatch": 100,
        "k":7,
        "iter":100,
        "lrate":1
    }

class LR:
    def __init__(self, dims) -> None:
        self.dims = dims + 1    # merge with b
        # self.W = np.random.normal(loc=0, scale=1, size=(self.dims, 1)) # of shape (1, dims)
        self.W = np.zeros((self.dims, 1))
        self.trainMode = True
    
    def eval(self):
        self.trainMode = False

    # def linearRegression(self, X, label:np.array):
    #     label = label.reshape((-1,1))
    #     label_mean = label.mean()
    #     label -= label_mean
    #     s = 0
    #     for i in range(self.dims-1):
    #         Xi_mean = X[:,i]- X[:, i].mean()
    #         self.W[i]= (Xi_mean).dot(label) / (Xi_mean).T.dot(Xi_mean)
    #         s += self.W[i]*X[:, i].mean()
    #     self.W[-1] = label_mean - s

    def train(self, X, label, lrate=0.01, iters=100, batchsize=100):
        X = np.append(X, np.ones((len(X),1)), axis=1)
        label = label.reshape((-1,1))
        trainset = DataLoader(np.append(X,label,axis=1))
        self.trainMode = True
        Loss = []
        for iter in range(iters):
            for miniX, minilabel in trainset.Batch(batchsize=batchsize):   # get minibatch data
                # Y = np.maximum(np.minimum(self.predict(miniX), 0.95), 0.05)
                Y = self.predict(miniX)         # get forward pass output (probability of intput)
                Loss.append(self.Loss(Y, minilabel))
                ac = ((Y > 0.5).astype('int') == minilabel).mean()
                dW = miniX.T.dot(Y-minilabel)   # dW = - X*(label - Y)
                dW /= len(miniX)
                self.W -= lrate*dW              # W -= lrate*dW
            
        return Loss
    
    def Fullnet(self, X):
        return X[:,:self.dims-1].dot(self.W[:-1]) + self.W[-1]

    def logistic(self,X):
        return 1/(1+np.exp(-X))
    
    def predict(self, datas:Iterable, label:Union[np.ndarray, None]=None):
        prob = self.logistic(self.Fullnet(datas))
        if self.trainMode:
            return prob
        else:
            predLable = (prob > 0.5).astype('int')
            if label is not None:
                label = label.reshape((-1,1))
                return predLable, (predLable == label).mean()
            else: return predLable
    
    def Loss(self, Y, label):
        # Y : predicted, label: Ground Truth
        return -(label.T.dot(np.log(Y))+(1-label).T.dot(np.log(1-Y))) / len(Y)
    
    # def predictSingle(self, data):
    #     return self.logistic(data[:self.dims].dot(self.W))


def main1():
    config = configFile()
    dataset = pd.read_csv(config['trainFile'], header=None).values
    dataset[:,-1] = dataset[:,-1].astype('int')
    KFloder = DataLoader(dataset)
    ac_lr = {}
    for lrate in [1e-3, 1e-2, 1e-1, 1]:
        total_ac = 0
        for trainset, trlabel, valset, valabel in KFloder.KfolderData(config["k"], shuffle=True):
            dims = len(trainset[0])
            lrMod = LR(dims)
            lrMod.train(trainset, label=trlabel, lrate=lrate)
            #lrMod.linearRegression(trainset[:,:-1],label=label)
            lrMod.eval()
            pred, ac = lrMod.predict(valset, valabel)
            total_ac += ac
            # result = np.append(valset, pred, axis=1)
            # pd.DataFrame(result, index=None, columns=None).to_csv(config['outputFile']+ 'ac:%f_.csv'% ac  , sep=',')
        total_ac /= config["k"]
        print("average ac of lrate:%f is %f" %(lrate, total_ac))
        ac_lr['lrate'] = total_ac

def main2():
    config = configFile()
    lrate=config['lrate']
    trainset = pd.read_csv("check/train_sample.csv", header=None).values
    valset =  pd.read_csv("check/test.csv", header=None).values
    ac_lr = {}
    total_ac = 0
    trlabel = trainset[:,-1].reshape(-1,1)
    trainset = trainset[:,:-1]
    valabel = valset[:,-1].reshape(-1,1)
    valset = valset[:,:-1]
    dims = len(trainset[0])
    lrMod = LR(dims)
    lrMod.train(trainset, label=trlabel, lrate=lrate)
    #lrMod.linearRegression(trainset[:,:-1],label=label)
    lrMod.eval()
    pred, ac = lrMod.predict(valset, valabel)
    result = np.append(valabel, pred, axis=1)
    pd.DataFrame(result, index=None, columns=None).to_csv(config['outputFile']+ 'ac:%f_.csv'% ac  , sep=',')
    print("ac of lrate:%f is %f" %(lrate, ac))


if __name__ == "__main__":
    main2()



# class LR_ML:
#     def __init__(self, dims) -> None:
#         self.dims = dims + 1    # merge with b
#         self.W = np.random.random((self.dims, 1)) # of shape (1, dims)
#         self.trainMode = True
    
#     def eval(self):
#         self.trainMode = False


#     def train(self, trainset, lrate=0.01, iters=10):
#         # 
#         for iter in range(iters):
#             # Y1 = self.W.dot(X)
#             Y1,X1 = self.Fullnet(trainset)
#             Y2,X2 = self.logistic(Y1)
#             loss, X3 = self.Loss(Y2, trainset[:,-1])
#             # backward
#             dy2 = self.backwardLoss()
#             dy1 = self.backwardLogi(Y2, dy2)
#             dW = self.backwardFullnet(X1, dy1)
#             self.W -= lrate*dW


#     def Fullnet(self, X):
#         return X[:,:self.dims].dot(self.W), X

#     def backwardFullnet(self, X, dy):
#         # return dW
#         return X.T.dot(dy)

#     def logistic(self,X):
#         return 1/(1+np.exp(X))

#     def backwardLogi(self, Y, dy):
#         # dy = -label*(1/Y) + (1-label)*(1/(1-Y))
#         # so, backwardLogi return -(label*(1-Y) - (1-label)*Y) = -(label-Y)
#         return Y*(1-Y)*dy
    
#     def Loss(self, X, Y):
#         return -(Y*np.log(X)+(1-Y)*np.log(1-X)), X

#     def backwardLoss(self, Y, label):
#         return -label*(1/Y) + (1-label)*(1/(1-Y))
    
#     def predictSingle(self, data):
#         return self.logistic(data[:self.dims].dot(self.W))

#     def predict(self, datas:Iterable):
#         return self.logistic(self.Fullnet(datas))