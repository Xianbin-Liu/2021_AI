from re import T
import sys
import numpy as np
from typing import Iterable
import matplotlib.pyplot as plt
import copy
from numpy.lib.shape_base import get_array_wrap

from numpy.random.mtrand import beta
sys.path.append("../")


class DataLoader:
    def __init__(self, datas:Iterable) -> None:
        self.datas = datas

    #数据分解
    def KfolderData(self, K:int, shuffle=False, test=False):
        '''
        @K：        K折交叉验证的K值
        @shuffle：  划分验证集的时候是否打乱数据
        @test：     True时将只返回一组训练集和验证集（用于调试，不需要进行K次）
        @return：   返回生成器：可用for进行迭代，每次产出：训练集数据、训练集标签（已矫正为N*1）、验证集数据、验证集标签（已矫正为N*1）
        '''
        # data should include labels
        if shuffle : np.random.shuffle(self.datas)
        numOfPart = int(len(self.datas)/K)
        if not test:
            for i in range(K):
                trainset = np.append(self.datas[numOfPart*(i+1):, :], self.datas[:numOfPart*i,:], axis=0)
                valset = self.datas[numOfPart*i:numOfPart*(i+1),:]
                yield trainset[:,:-1], trainset[:,-1].reshape(-1,1), valset[:,:-1], valset[:,-1].reshape(-1,1)
        else:
            trainset, valset = self.datas[numOfPart:], self.datas[:numOfPart]
            for i in range(1):
                yield trainset[:,:-1], trainset[:,-1].reshape(-1,1), valset[:,:-1], valset[:,-1].reshape(-1,1)
    
    # ！！不需要显式调用！！
    def Batch(self, batchsize=100, shuffle=True, throw=True):
        '''
        @batchsize: 批大小
        @shuffle：  是否打乱数据
        @throw：    是否丢弃最后一组（大小不满足batchsize）
        @return：   训练数据、标签（已矫正为N*1）
        '''
        # with label
        if shuffle: np.random.shuffle(self.datas)
        n = len(self.datas)/batchsize
        n = int(n) if throw else np.ceil(n) # whether throw away the last set

        for i in range(n):
            yield self.datas[batchsize*i:batchsize*(i+1),:-1], self.datas[batchsize*i:batchsize*(i+1),-1].reshape(-1,1)
    

class LrateStrategy:
    '''
    	学习率策略：暂未实现
    '''
    def __init__(self, W, b, lr=1e-2) -> None:
        self.lr = lr
        self.moment = lr
        self.t = 0
        self.gW = []
        self.gb = []
        self.gW2 = []
        self.gb2 = []
        for w in W:
            self.gW.append(np.zeros(w.shape))
        
        for eb in b:
            self.gb.append(np.zeros(eb.shape))


    def warmup(self):
        pass
    
    def Adagrad(self):
        def adagrad(dW, db):
            self.t += 1
            for i in range(len(dW)):
                self.gW[i] = np.sqrt((np.square(self.gW[i])*(self.t-1) + np.square(dW[i]))/self.t)
                self.gb[i] = np.sqrt((np.square(self.gb[i])*(self.t-1) + np.square(db[i]))/self.t)
            gW = copy.deepcopy(self.gW)
            gb = copy.deepcopy(self.gb)
            for i in range(len(gW)):
                gW[i] = dW[i] / gW[i]
                gb[i] = db[i] / gb[i]
            return gW, gb
        
        return adagrad

    def Adam(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.gW2 = copy.deepcopy(self.gW)
        self.gb2 = copy.deepcopy(self.gb)
        
        def adam(dW, db):
            self.t += 1
            gW = copy.deepcopy(self.gW)
            gb = copy.deepcopy(self.gb)
            for i in range(len(dW)):
                # momentum
                self.gW[i] = beta1*self.gW[i] + (1-beta1)*dW[i]
                self.gb[i] = beta1*self.gb[i] + (1-beta1)*db[i]
                # rmsprop
                self.gW2[i] = beta2*self.gW2[i] + (1-beta2)*np.square(dW[i])
                self.gb2[i] = beta2*self.gb2[i] + (1-beta2)*np.square(db[i])
                
                gW[i] = self.gW[i] / (1-pow(beta1, self.t))
                gW[i] = gW[i] / ( np.sqrt(self.gW2[i]/(1-beta1**self.t)) +epsilon)
                
                gb[i] = self.gb[i] / (1-pow(beta1, self.t))
                gb[i] = gb[i] / ( np.sqrt(self.gb2[i]/(1-beta2**self.t)) +epsilon)
            
            return gW, gb
        return adam
    
    # dont use
    def RMSprop(self, alpha=0.1, eplison=1e-8):
        def RMSpropAlpha(dW, db):
            self.t += 1
            gW = copy.deepcopy(self.gW)
            gb = copy.deepcopy(self.gb)
            for i in range(len(dW)):
                self.gW[i] = alpha*np.square(self.gW[i])+(1-alpha)*np.square(dW[i])
                gW[i] = dW[i] / (np.sqrt(self.gW[i])+eplison)
                self.gb[i] = alpha*np.square(self.gb[i])+(1-alpha)*np.square(db[i])
                gb[i] = db[i] / (np.sqrt(self.gb[i])+eplison)

            return gW, gb
        return RMSpropAlpha


def plotTraining(Loss, ac_t, ac_v=[]):
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

def log(text:str="", config:dict[str,str]={}, file="log/classification/log.txt"):
    origin_stdo = sys.stdout
    with open(file=file, mode="a") as f:
        sys.stdout = f
        print("\n", text)
        for key, val in config.items():
            print(key,":",val)
    sys.stdout = origin_stdo