from enum import Flag
from os import lstat
from types import LambdaType
from nltk.grammar import _read_dependency_production
import numpy as np
from numpy.core.fromnumeric import trace
from numpy.core.numeric import ones
from numpy.lib.function_base import blackman
from numpy.lib.twodim_base import tril, triu_indices_from
from numpy.random.bit_generator import SeedlessSeedSequence
from numpy.random.mtrand import gamma, set_state
import pandas as pd
from typing import Iterable, List, Union
import matplotlib.pyplot as plt
import sys
import re
from pandas.core.algorithms import isin
from torch._C import _debug_set_autodiff_subgraph_inlining, set_flush_denormal

from torch.nn.modules import batchnorm, loss
from torch.utils import data

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
        "epoch":250,
        "lrate":0.1
    }
    
# use to debug with one layer : y = Wx+b
class NerualNet:
    def __init__(self, paramsFile=None, featureDim=None, outputDim=1, HiddenDims=[], lastACT="sigmod", weight_scale=1e-2) -> None:
        '''
        @paramsFile:    参数文件，如果指定了该参数文件的路径，将从该文件重建参数，而不是随机初始化；直接指出文件名即可，不需要添加后缀（如需添加后缀，请保证后缀为.npz）
        @featureDim:    输入的特征维数
        @outputDim:     输出的特征维数，如二分类或者回归预测的时候，都为1
        @HiddenDims:    中间层，请保证中间层是一个列表，列表的元素依次为每一层中间层的输入维数
        @weight_scale:  初始化特征权重，一般默认是1e-2即可，用于保证用np.random初始化的参数不要过大
        '''
        self.lastACT = lastACT
        self.layers = self.dims = 0
        self.W = []
        self.b = []
        self.BNParam = []

        # 初始化权重：正态分布（0，1）
        if paramsFile != None:
           self.loadParam(paramsFile)
           self.dims = len(self.W[0])
           HiddenDims = [self.dims]
           for i in range(self.layers):
               HiddenDims.append(self.W[i].shape[1])
        else:
            self.dims = featureDim   # merge with b
            HiddenDims.insert(0, featureDim)
            HiddenDims.append(outputDim)
            for i in range(len(HiddenDims)-1):
                self.W.append((np.random.randn(HiddenDims[i], HiddenDims[i+1]) / np.sqrt(HiddenDims[i])/2).astype('float64')) #dont merge
                self.b.append(np.random.randn(1, HiddenDims[i+1]).astype('float64'))
            #self.W = np.random.random((self.dims, 1)) # of shape (1, dims)
        
        # Batch Norm 
        for i in range(len(HiddenDims)-1):
            self.BNParam.append( {'gamma': np.random.uniform(0, 1, (1, HiddenDims[i+1])), 'beta':np.zeros((1, HiddenDims[i+1])), 'running_mean': np.zeros((1, HiddenDims[i+1])), 'running_var':np.zeros((1, HiddenDims[i+1]))})

        self.layers = len(self.W)
        self.trainMode = True
        self.groupReLU = (1, 0.1)
        self.activation = []
        self.backwardactivation = []


    @property
    def param(self):
        '''
        @return:    返回参数字典，如：param['W1']取出W1的参数矩阵
        '''
        param = {}
        for i in range(self.layers):
            param["W"+str(i+1)] = self.W[i]
            param["b"+str(i+1)] = self.b[i]
        return param

    def eval(self):
        # 使模型关闭训练状态，请确保在预测测试集时，将训练状态关闭
        self.trainMode = False

    def trainning(self):
        # 打开训练状态，请确保在训练前，打开训练状态
        self.trainMode = True

    def loadParam(self, file):
        # 加载参数文件，file为路径，默认文件结构为
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
        # 保存参数到指定文件路径，默认后缀为.npz
        np.savez(file, **self.param)
        
    
    def addActivation(self, name:Union[str, list[str]]):
        if isinstance(name, str):
            name = name.lower()
            if name == 'sigmod':
                self.activation.append(self.sigmod)
                self.backwardactivation.append(self.backwardSigmod)
            elif name == 'ReLU':
                self.activation.append(self.ReLU)
                self.backwardactivation.append(self.backwardReLU)

            elif name == 'linear':
                self.activation.append(None)
                self.backwardactivation.append(self.backwardReLU)

            self.lastACT = name

        
        else:
            for n in name:
                n = n.lower()
                if n == 'sigmod':
                    self.activation.append(self.sigmod)
                    self.backwardactivation.append(self.backwardSigmod)
                elif n == 'relu':
                    self.activation.append(self.ReLU)
                    self.backwardactivation.append(self.backwardReLU)
                elif n == 'linear':
                    self.activation.append(None)
                    self.backwardactivation.append(self.backwardReLU)
                self.lastACT = n

    def Loss(self, X, Labels=None, epsilonmax=0.95, epsilonmin=0.05, lossfunction="RMSELoss", lamda=0, bn=False, bn_tr=True):
        '''
        @intro：        ！！！不需要外部调用！！！
        @X:             输入X
        @Labels：       真实值
        @eplision：  保证传入MLEloss函数的输入的上下限
        @lossfunction： 指定损失函数，目前版本只支持：MLELoss(用于二分类)，RMSELoss（均方根误差：用于回归预测）, MSELoss
        '''
        # Y1 = self.W.dot(X)
        dW_all, db_all, X_fullnet = [],[],[]    # 暂存中间值： 
        datas = []
        batchnormDatas  = []
        dgamma_all, dbeta_all = [],[]
        Y = None  # predict

        # forward
        for i in range(self.layers):
            data = {}
            Y,X = self.Fullnet(X,self.W[i],self.b[i])
            X_fullnet.append(X)

            if bn:  # batch norm
                bndata = self.BatchNorm(Y, self.BNParam[i], train=bn_tr)
                batchnormDatas.append(bndata)
                #if i != self.layers -1: 
                    # dont use in last layer : clear out the layers ....> # no it's wrong, it's reason is that batchsize set to 1
                    
                Y = bndata['Y']

            acfunc = self.activation[i]
            if acfunc != 'linear' and acfunc != None:  # 最后一层激活函数如果存在的话
                data = acfunc(Y) #  data保存了激活层的输入和输出，用于反向传播  ：Y,X = sigmod(X) 

            else:
                data['X']=Y
                data['Y']=Y
            X = data['Y']   #  Next layer input
            datas.append(copy.deepcopy(data))
            del data

        # last layer output
        data = datas[-1]

        # # here: Y is the last activation result, now change it with parameter self.lastACT
        # if self.lastACT == "ReLU":
        #     datas[-1] = self.ReLU(data['X'])
        # elif self.lastACT == "linear":
        #     datas[-1] = data['X']
        # elif self.lastACT == 'sigmod':
        #     datas[-1] = self.sigmod(data['X'])

        if Labels is None:
            return datas[-1]['Y']

        # perform Loss
        if lossfunction == "MLELoss":
            datas[-1]['Y'] = np.minimum(np.maximum(datas[-1]['Y'], epsilonmin), epsilonmax)   # make Y in (0,1) instead :[0,1]
            lossf = self.MLELoss
            backSigLoss = self.backwardSigmodMLELoss
            if self.lastACT == "ReLU":
                backSigLoss = self.backwardReLUMLELoss

            elif self.lastACT == "linear":
                backSigLoss = self.backwardMLELoss

        elif lossfunction == "RMSELoss":
            lossf = self.RMSELoss
            backSigLoss = self.backwardSigmodRMSE
            if self.lastACT == "ReLU":
                backSigLoss = self.backwardReLURMSE
            elif self.lastACT == "linear":
                backSigLoss = self.backwardRMSE
            
        elif lossfunction == "MSELoss":
            lossf = self.MSEloss
            backSigLoss = self.backwardSimgodMSE
            if self.lastACT == "ReLU":
                backSigLoss = self.backwardReLUMSE
            elif self.lastACT == "linear":
                backSigLoss = self.backwardMSE


        else:
            raise ValueError("请确定lossFunction的名称")
        
        # regularization
        W_square = 0
        for i in range(self.layers):
            W_square += self.W[i].flatten().dot(self.W[i].flatten().T) + self.b[i].flatten().dot(self.b[i].flatten().T)


        # datas[-1] struct : {'X':最后激活层输入, 'Y'；预测值, 'loss':loss} 
        datas[-1] = lossf(datas[-1], Y=Labels)
        datas[-1]['loss'] += lamda*W_square
        if not self.trainMode:
            return datas[-1]['Y'], datas[-1]['loss']

        # last layer backard with Loss
        dy = backSigLoss(datas[-1], Y=Labels)
        if bn:
            if self.lastACT != 'linear':
                cache = self.backwardBatchNorm(batchnormDatas[-1], dy)
                dgamma, dbeta, dy = cache['dgamma'], cache['dbeta'], cache['dx']
                dgamma_all.insert(0, dgamma)
                dbeta_all.insert(0, dbeta)
            else:
                dgamma_all.insert(0, np.zeros((1, self.W[-1].shape[1])))
                dbeta_all.insert(0, np.zeros((1, self.W[-1].shape[1])))

        
            
        dW, db, dy = self.backwardFullnet(self.W[-1], X_fullnet[-1], dy)

        dW_all.insert(0, dW)
        db_all.insert(0, db)
        
        for i in range(self.layers-2, -1, -1):
            dy = self.backwardactivation[i](datas[i], dy_upper=dy)
            if bn:
                cache = self.backwardBatchNorm(batchnormDatas[i], dy)
                dgamma, dbeta, dy = cache['dgamma'], cache['dbeta'], cache['dx']
                dgamma_all.insert(0, dgamma)
                dbeta_all.insert(0, dbeta)
            dW, db, dy = self.backwardFullnet(self.W[i], X_fullnet[i], dy)
            dW_all.insert(0, dW)
            db_all.insert(0, db)

        return {"dW":dW_all, "db":db_all, "loss":datas[-1]['loss'], 'dgamma':dgamma_all, 'dbeta':dbeta_all}

    def train(self, X, labels=None, valSet=None, valabel=None, lrate=0.01, epochs=10, batchSize=0, lossfunction="RMSELoss", lamda=0, optim=None, bn=False, bn_tr=True):
        '''
        @X：        请确保输入的X不带有标签
        @labels：   输入的标签
        @valset：   验证集：    如果指定的话，则标签也应该给出，会在每10次迭代中对验证集进行预测和评估
        @valset：   验证集标签
        @lrate：    学习率
        @epochs：   迭代次数： 完整遍历一次训练集的次数
        @batchSize：批次大小：每次迭代的训练集大小；指定为0时，迭代大小为整个训练集大小（批梯度）
        @lossfunction： RMSELoss（均方根误差：回归），MLELoss（似然误差：二分类）
        @return：   Loss（每10次迭代进行一次记录），ac_t（训练集准确率，每10次迭代记录一次），ac_v(验证集准确率：每10次迭代记录一次)     
        '''
        
        # ensure X NOT append with label
        if batchSize == 0: batchSize = len(X)
        Loss = []
        Loss_x = []
        Loss_v = []
        dW_2 = []
        cnt = 0

        for epoch in range(epochs):
            if labels is None: labels =np.ones((len(X),1))
            for miniX, minY in DataLoader(np.append(X,labels, axis=1)).Batch(batchSize,shuffle=True):
                cnt += 1
                cache = self.Loss(miniX, minY, lossfunction=lossfunction, lamda=lamda, bn=bn, bn_tr=bn_tr)
                dW, db, loss, dgamma, dbeta = cache["dW"], cache["db"], cache["loss"], cache['dgamma'], cache['dbeta']
                
                if optim is not None:
                    dW, db = optim(dW, db)
                for i in range(self.layers):
                    self.W[i] -= lrate*(dW[i]+lamda*self.W[i])
                    self.b[i] -= lrate*(db[i]+lamda*self.b[i])
                    if bn:
                        self.BNParam[i]['gamma'] -= lrate*(dgamma[i])
                        self.BNParam[i]['beta'] -= lrate*(dbeta[i])

                if cnt % 10 == 0:
                    Loss.append(loss.flatten())
                    dW_2.append((dW[0]*dW[0]).sum())
                    # ac = ((self.Loss(miniX)>0.5).astype('int').reshape(-1,1)==minY).mean()
                    # ac_t.append(ac)
                    # if valSet is not None:
                    #     ac = ((self.Loss(valSet)>0.5).astype('int').reshape(-1,1)==valabel).mean()
                    #     ac_v.append(ac)
                    Loss_x.append(self.Loss(X,labels, lossfunction=lossfunction, bn=bn, bn_tr=False)["loss"].flatten())
                    if valSet is not None:
                        Loss_v.append(self.Loss(valSet,valabel, lossfunction=lossfunction, bn=bn, bn_tr=False)["loss"].flatten())
        return Loss, Loss_x, Loss_v

    # return Y, X
    def Fullnet(self, X, W, b):
        return X.dot(W)+b, X

    '''
    #   带有backward的指出这个函数是用于计算反向传播
    '''

    def backwardFullnet(self, W, X, dy_upper):
        # return dW AND dy
        dW = X.T.dot(dy_upper) / len(X)
        db = np.ones((1,len(X))).dot(dy_upper) / len(X)
        dy = dy_upper.dot(W.T)
        return dW, db, dy

    def BatchNorm(self, X, BNP, epsilon=1e-8, moment=0.9, train=True):
        gamma, beta = BNP['gamma'], BNP['beta']
        running_mean, running_var = BNP['running_mean'], BNP['running_var']
        m = len(X)
        if train:
            miu = 1. / m * np.sum(X, axis=0)
            #BNP['miu'] += [miu]
            miu = running_mean * moment + (1-moment)*miu
            BNP['running_mean'] = miu
        else:
            miu = running_mean

        xmiu = X - miu
    

        sq = xmiu ** 2
        if train:
            var = 1./m * np.sum(sq, axis = 0)
            #BNP['var'] += [var]
            var = running_var * moment + (1-moment)*var
            BNP['running_var'] = var

        else:
            var = running_var

        
        sqr_var = np.sqrt(var + epsilon)
        i_var = 1. /sqr_var
        x_hat = xmiu / sqr_var
        return {'Y':x_hat * gamma + beta, 'gamma':gamma, 'beta':beta, 'xmiu':xmiu, 'i_var':i_var, 'sqr_var':sqr_var, 'var':var, 'epsilon':epsilon, 'x_hat':x_hat}

    def backwardBatchNorm(self, data, dy_upper=0):
        Y, gamma, beta, xmiu, i_var, sqr_var, var, epsilon = data['Y'], data['gamma'], data['beta'], data['xmiu'], data['i_var'], data['sqr_var'], data['var'], data['epsilon']
        x_hat = data['x_hat']
        N,D = dy_upper.shape
        dbeta = np.ones((1, N)).dot(dy_upper)
        dgamma_x = dy_upper

        dgamma = np.ones((1,N)).dot(dy_upper*x_hat)
        dx_hat = dgamma_x * gamma

        dx_miu = dx_hat * i_var
        di_var = np.ones((1, N)).dot(dx_hat*xmiu)

        dsqr_var = di_var * (-1. / sqr_var**2)

        dvar = 0.5 * 1. / np.sqrt(var+epsilon) * dsqr_var

        dsq = 1./N * np.ones((N,D)) * dvar

        dx_miu2 = 2*xmiu*dsq

        dx1 = (dx_miu + dx_miu2)
        dmiu = -1 * np.ones((1, N)).dot(dx_miu + dx_miu2)

        dx2 = 1. / N * np.ones((N,D)) * dmiu

        dx = dx1 + dx2

        return {'dx': dx,  'dgamma':dgamma, 'dbeta':dbeta}


    def sigmod(self,X):
        return {'Y':1/(1+np.exp(-X)), 'X':X}

    # def backwardSigmodFullnet(self, X=0, Y=0, dy_upper=0):
    #     pass

    def backwardSigmodMLELoss(self, data, Y):
        Yp = data['Y']
        return -(Y-Yp)


    def ReLU(self, X):
        return {'Y':(X*(X>0)*self.groupReLU[0] + X*(X<0)*self.groupReLU[1]).astype('float64'), 'X':X}

    def backwardReLUMLE(self, data, Y=0):
        dy = self.backwardMLELoss(data, Y)
        dy = self.backwardReLU(data, dy_upper=dy)
        return dy

    def backwardReLURMSE(self, data, Y=0):
        dy = self.backwardRMSE(data, Y)
        dy = self.backwardReLU(data, dy_upper=dy)
        return dy

    def backwardReLUMSE(self, data, Y=0):
        dy = self.backwardMSE(data, Y)
        dy = self.backwardReLU(data, dy)
        return dy

    # no Param
    def backwardReLU(self, data, dy_upper=0):
        # dy is upper derivate
        # return d(ReLu_y)/d(ReLU_x)
        X = data['X']
        return ((X > 0)*self.groupReLU[0] + (X<0)*self.groupReLU[1])*(dy_upper)


    def backwardSigmod(self, data, dy_upper=0):
        # X not used
        # dy = -label*(1/Y) + (1-label)*(1/(1-Y))
        # so, backwardLogi return -(label*(1-Y) - (1-label)*Y) = -(label-Y)
        Y = data['Y']
        return Y*(1-Y)*(dy_upper)
    
    # end Point
    def MLELoss(self, data, Y):
        # Y是标签

        Yp = data['Y']
        loss = -(Y.T.dot(np.log(Yp))+(1-Y).T.dot(np.log(1-Yp))) / len(Yp)
        data['loss'] = loss
        return data

    # end Point
    def backwardMLELoss(self, data, Y):
        # Y是标签
        Yp = data['Y']  # 预测值
        return -Y*(1/Yp) + (1-Y)*(1/(1-Yp))
    
    def MSEloss(self, data, Y):
        Yp = data['Y']   # 预测值
        loss = (Y.T.dot(Y)+Yp.T.dot(Yp)-2*Y.T.dot(Yp))/len(Y)
        data['loss'] = loss
        return data
    
    def backwardMSE(self, data, Y):
        Yp = data['Y']
        return 2*(Yp-Y)

    def RMSELoss(self, data, Y):
        Yp = data['Y']
        loss = np.sqrt((Y.T.dot(Y)+Yp.T.dot(Yp)-2*Y.T.dot(Yp))/len(Y))
        data['loss'] = loss
        return data

    def backwardRMSE(self, data, Y):
        Yp = data['Y']
        Loss = data['loss']
        return (Yp-Y)/Loss
    
    def backwardSigmodRMSE(self, data, Y):
        
        dy = self.backwardRMSE(data, Y=Y)
        dy = self.backwardSigmod(data, dy_upper=dy)
        return dy

    def backwardSimgodMSE(self, data, Y):
        dy = self.backwardMSE(data, Y)
        dy = self.backwardSigmod(data, dy_upper=dy)
        return dy


    # 预测
    def predict(self, datas:Iterable, labels=None, threshold=0.5, ac=False, bn=False):
        '''
        @data:      输入数据，请确保没有label
        @labels:    默认为None：仅输出神经网络的预测值；如果指定了labels，按照阈值将预测值转换为类别，并计算准确率    
        @threshold：阈值
        @return：   预测值（以及准确率，如果指定了labels的话）
        '''
        # fit
        self.eval()
        if labels is not None:
            res, Loss = self.Loss(datas, labels, bn=bn, bn_tr=False)
        else: return self.Loss(datas, bn=bn, bn_tr=False)
        if ac:
            res = (res > threshold).astype('int').reshape((-1, 1))
            return res, (res==labels).mean()
        else: return res, Loss

def main2():
    config = configFile()
    lrate=config['lrate']
    dataset = pd.read_csv("check/train.csv", header=None).values
    dataset[:,-1] = dataset[:,-1].astype('int')
    loader = DataLoader(dataset)
    for trainset, trlabel, valset, valabel in loader.KfolderData(7, shuffle=True, test=True):
        dims = len(trainset[0])
        lrMod = NerualNet(featureDim=dims, outputDim=1, HiddenDims=[40])
        strategy = LrateStrategy(lrMod.W, lrMod.b)
        adagrad = strategy.Adagrad()
        Loss, ac_t, ac_v = lrMod.train(trainset, labels=trlabel, lrate=lrate, epochs=config["epoch"], batchSize=1000, lossfunction="MLELoss", lamda=0.0, optim=adagrad)
        plotTraining(Loss, ac_t, ac_v)

        # plt.show()
        lrMod.eval()
        pred, ac = lrMod.predict(valset, valabel, ac=True)
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
    