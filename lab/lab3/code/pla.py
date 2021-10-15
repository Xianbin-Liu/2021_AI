import numpy as np
import pandas as pd
import sys
from typing import Union, Iterable
from pandas.core.frame import DataFrame
sys.path.append('../')

def sign(x:Union[int,float]):
    return 2*int(x > 0)-1

def signV(X:Union[np.ndarray, pd.DataFrame]):
    return 2*(X>0).astype('int')-1


def configFile():
    return {
        "trainFile":"data/train.csv",
        "outputFile":"check/18308133_liuxianbin_pla",
        "trainMode":1,
        "vrate":0.25,
        "k":7,
        "lrate":1
    }

class DataLoader:
    def __init__(self, datas:Iterable) -> None:
        self.datas = datas
    
    def KfolderData(self, K:int, shuffle=False):
        if shuffle : np.random.shuffle(self.datas)
        numOfPart = int(len(self.datas)/K)
        for i in range(K):
            trainset = np.append(self.datas[numOfPart*(i+1):, :], self.datas[:numOfPart*i,:], axis=0)
            valset = self.datas[numOfPart*i:numOfPart*(i+1),:]
            yield trainset[:,:-1], trainset[:,-1], valset[:,:-1], valset[:,-1]

class PLA: 
    def __init__(self, dims) -> None:
        # merge W and 
        self.dims = dims + 1
        # shape W,b -> [W, b] in column vector
        # self.W = np.random.random((self.dims, 1))
        self.W  = np.zeros((self.dims, 1))
    
    def train(self, trainset, labels, lrate=1, iters=100, error_rate=0):
        # SGD
        # trainset = np.insert(trainset, self.dims-1, 1, axis=1)
        # for iter in range(iters):
        #     for epoch in range(len(trainset)):
        #         # update for each data
        #         data = trainset[epoch]
        #         label= labels[epoch]
        #         while self.predSingle(data) != 2*label-1:       # convert{0,1}->{-1,1} using map: y=2x-1
        #             self.W += (lrate*(2*label-1)*data[:self.dims]).reshape(-1,1)
        iter = 0
        while (iter < iters):
            error_num = 0       # record the misclassification times over each look of the whole dataset
            # training on all dataset
            for epoch in range(len(trainset)):
                data, label = trainset[epoch], labels[epoch] # get one single data
                if iter >= iters:
                    break
                # if misclassification-->update iteration times, error times and W
                if self.predSingle(data) != 2*label-1:       # convert{0,1}->{-1,1} using map: y=2x-1
                    iter += 1
                    error_num += 1
                    # dW = - y_i*x_i
                    self.W += (lrate*(2*label-1)*data[:self.dims]).reshape(-1,1)
            # while error_rate in trainset smaller than given threshold, BREAK
            if error_num/len(trainset) <= error_rate : break 
    
    def predSingle(self, data):
        return sign(data[:self.dims].dot(self.W))

    def predict(self, datas):
        return signV(datas[:,:self.dims].dot(self.W[:-1])+self.W[-1])

def main():
    config = configFile()
    dataset = pd.read_csv(config['trainFile'], header=None).values
    dataset[:,-1] = dataset[:,-1].astype('int')
    Dataset = DataLoader(dataset)

    i = 1
    for lrate in [1e-3, 1e-2, 1e-1, 1]:
        total_ac = 0
        for trainset, trainlabel, valset, valabel in  Dataset.KfolderData(config['k'], shuffle=True):
            valabel = valabel.reshape((-1,1))
            dims = len(trainset[0])
            trainset = np.append(trainset, np.ones((len(trainset), 1)), axis=1)
            plaMod = PLA(dims)
            plaMod.train(trainset, trainlabel, lrate=lrate)
            pred = plaMod.predict(valset)
            ac = (pred == 2*valabel-1).mean()
            # print("the ac with floder %d / %d is %f"%(i, config["k"], ac))
            i += 1
            total_ac += ac
        total_ac /= config["k"]
        print("the average ac with lrate %f is %f " %(lrate, total_ac))


def main2():
    config = configFile()
    lrate=config['lrate']
    trainset = pd.read_csv("check/train_sample.csv", header=None).values
    valset =  pd.read_csv("check/test.csv", header=None).values
    ac_lr = {}
    total_ac = 0
    trlabel = trainset[:,-1].reshape(-1,1)
    trainset = trainset[:,:-1]
    dims = len(trainset[0])
    trainset = np.append(trainset, np.ones((len(trainset),1)), axis=1)
    valabel = valset[:,-1].reshape(-1,1)
    valset = valset[:,:-1]
    PlaMod = PLA(dims)
    PlaMod.train(trainset, labels=trlabel, lrate=lrate)
    #lrMod.linearRegression(trainset[:,:-1],label=label)
    pred = PlaMod.predict(valset)
    valabel = 2*valabel-1
    ac = (pred == valabel).mean()
    result = np.append(valabel, pred, axis=1)
    pd.DataFrame(result, index=None, columns=None).to_csv(config['outputFile']+ 'ac:%f_.csv'% ac  , sep=',')
    print("ac of lrate:%f is %f" %(lrate, ac))
if __name__ == '__main__':
    main2()