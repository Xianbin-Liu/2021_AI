import numpy as np
from typing import Iterable

class DataLoader:
    def __init__(self, datas:Iterable) -> None:
        self.datas = datas

    def KfolderData(self, K:int, shuffle=False):
        # data should include labels
        if shuffle : np.random.shuffle(self.datas)
        numOfPart = int(len(self.datas)/K)
        for i in range(K):
            trainset = np.append(self.datas[numOfPart*(i+1):, :], self.datas[:numOfPart*i,:], axis=0)
            valset = self.datas[numOfPart*i:numOfPart*(i+1),:]
            yield trainset[:,:-1], trainset[:,-1].reshape(-1,1), valset[:,:-1], valset[:,-1].reshape(-1,1)
    
    def Batch(self, batchsize=100, shuffle=True, throw=True):
        # with label
        if shuffle: np.random.shuffle(self.datas)
        n = len(self.datas)/batchsize
        n = int(n) if throw else np.ceil(n) # whether throw away the last set

        for i in range(n):
            yield self.datas[batchsize*i:batchsize*(i+1),:-1], self.datas[batchsize*i:batchsize*(i+1),-1].reshape(-1,1)
    

class LrateStrategy:
    def __init__(self, lr=1e-2) -> None:
        self.lr = lr
    
    def warmup(self):
        pass