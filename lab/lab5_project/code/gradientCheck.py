import numpy as np
import sys
from pandas.core.algorithms import mode
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.modules import adaptive
from torch.nn.modules.container import ModuleList
import torch.optim as optim
sys.path.append('../')
sys.path.append('../code')
from util import *
from NeuralNet import *

# sigmod/backward check
def gcheck1(model:NerualNet, X):
    Y1, X = model.sigmod(X)
    dy1 = model.backwardSigmod(X, Y1, dy_upper=1)
    
    Xtorch = torch.FloatTensor(X)
    Xtorch = Variable(Xtorch, requires_grad=True)
    y = torch.sigmoid(Xtorch)
    y.backward(gradient=torch.ones(X.shape))
    print(abs((dy1-Xtorch.grad.numpy())<1e-7).all())

# fullnet / backward check, MSE/ backward check, MSEsigmod backward check
def gcheck2(model:NerualNet, X, Y):
    W = np.random.random((30, 1))
    b = np.random.random(1).reshape((1,-1))
    Y0, X0 = model.Fullnet(X, W, b.reshape((1,-1)))
    Y1, X1 = model.sigmod(Y0)
    loss = model.MSEloss(Y1, Y)

    #dy1 = model.backwardMSE(Y1, Y, Loss=loss)
    dy2 = model.backwardSimgodMSE(Yp=Y1, Y=Y, Loss=loss)
    dW, db, dy3=  model.backwardFullnet(W, X, dy2)
    
    Xtorch = torch.FloatTensor(X)
    Wtorch = torch.FloatTensor(W)
    btorch = torch.FloatTensor(b)
    Xtorch = Variable(Xtorch, requires_grad=True)
    Wtorch = Variable(Wtorch, requires_grad=True)
    btorch = Variable(btorch, requires_grad=True)
    
    y01 = torch.mm(Xtorch, Wtorch)
    y0 = y01 + btorch
    y1 = torch.sigmoid(y0)
    Losf = nn.MSELoss()
    y2 = Losf(y1, torch.FloatTensor(Y))
    y2.backward()
    print(abs((dy3/100-Xtorch.grad.numpy())<1e-7).all())
    print((abs(dW-Wtorch.grad.numpy())<1e-7).all())
    print((abs(db-btorch.grad.numpy())<1e-7).all())

# MLELoss,  sigmodMLELoss backward check
def gcheck3(model:NerualNet, X, Y):
    W = np.random.random((30, 1))
    b = np.random.random(1).reshape((1,-1))
    Y0, X0 = model.Fullnet(X, W, b.reshape((1,-1)))
    Y1, X1 = model.sigmod(Y0)
    loss = model.MLELoss(Y1, Y)

    #dy1 = model.backwardMSE(Y1, Y, Loss=loss)
    dy2 = model.backwardSigmodMLELoss(Yp=Y1, Y=Y, Loss=loss)
    dW, db, dy3=  model.backwardFullnet(W, X, dy2)
    
    Xtorch = torch.FloatTensor(X)
    Wtorch = torch.FloatTensor(W)
    btorch = torch.FloatTensor(b)
    Xtorch = Variable(Xtorch, requires_grad=True)
    Wtorch = Variable(Wtorch, requires_grad=True)
    btorch = Variable(btorch, requires_grad=True)
    
    y01 = torch.mm(Xtorch, Wtorch)
    y0 = y01 + btorch
    y1 = torch.sigmoid(y0)
    Losf = nn.BCELoss()
    y2 = Losf(y1, torch.FloatTensor(Y))
    y2.backward()
    print(abs((dy3/100-Xtorch.grad.numpy())<1e-7).all())
    print((abs(dW-Wtorch.grad.numpy())<1e-7).all())
    print((abs(db-btorch.grad.numpy())<1e-7).all())


# Multi Layer check with MLELoss
def gcheck4(X, Y):
    W1 = np.random.randn(30, 40) * 1e-1
    b1 = np.random.randn(40).reshape((1,-1)) * 1e-1
    W2 = np.random.randn(40, 1) * 1e-1
    b2 = np.random.randn(1).reshape((1,-1)) * 1e-1
    model = NerualNet(featureDim=30, outputDim=1, HiddenDims=[40])
    model.W = [W1, W2]
    model.b = [b1, b2]
    cahe = model.Loss(X, Y, lossfunction='MLELoss')
    dW, db, Loss = cahe['dW'], cahe['db'], cahe['loss']
    Y0 = model.predict(X)

    Xtorch = torch.FloatTensor(X)
    W1torch = torch.FloatTensor(W1)
    b1torch = torch.FloatTensor(b1)
    Xtorch = Variable(Xtorch, requires_grad=True)
    W1torch = Variable(W1torch, requires_grad=True)
    b1torch = Variable(b1torch, requires_grad=True)
    W2torch = torch.FloatTensor(W2)
    b2torch = torch.FloatTensor(b2)
    W2torch = Variable(W2torch, requires_grad=True)
    b2torch = Variable(b2torch, requires_grad=True)
    
    y01 = torch.mm(Xtorch, W1torch)
    y0 = y01 + b1torch
    y1 = torch.sigmoid(y0)
    
    y21 = torch.mm(y1, W2torch)
    y2 = y21 + b2torch
    y3 = torch.sigmoid(y2)

    losf = nn.BCELoss()
    loss = losf(y3, torch.FloatTensor(Y))
    loss.backward()
    
    #print(abs((dy3/100-Xtorch.grad.numpy())<1e-7).all())
    print((abs(dW[0]-W1torch.grad.numpy())<1e-7).all())
    print((abs(db[0]-b1torch.grad.numpy())<1e-7).all())
    print((abs(dW[1]-W2torch.grad.numpy())<1e-7).all())
    print((abs(db[1]-b2torch.grad.numpy())<1e-7).all())


# Multi Layer check with MLELoss
def gcheck5(X, Y):
    W1 = np.random.randn(30, 40) * 1e-1
    b1 = np.random.randn(40).reshape((1,-1)) * 1e-1
    W2 = np.random.randn(40, 1) * 1e-1
    b2 = np.random.randn(1).reshape((1,-1)) * 1e-1
    model = NerualNet(featureDim=30, outputDim=1, HiddenDims=[40])
    model.W = [W1, W2]
    model.b = [b1, b2]
    cahe = model.Loss(X, Y, lossfunction='MLELoss')
    dW, db, Loss = cahe['dW'], cahe['db'], cahe['loss']
    Y0 = model.predict(X)

    Xtorch = torch.FloatTensor(X)
    W1torch = torch.FloatTensor(W1)
    b1torch = torch.FloatTensor(b1)
    Xtorch = Variable(Xtorch, requires_grad=True)
    W1torch = Variable(W1torch, requires_grad=True)
    b1torch = Variable(b1torch, requires_grad=True)
    W2torch = torch.FloatTensor(W2)
    b2torch = torch.FloatTensor(b2)
    W2torch = Variable(W2torch, requires_grad=True)
    b2torch = Variable(b2torch, requires_grad=True)
    
    y01 = torch.mm(Xtorch, W1torch)
    y0 = y01 + b1torch
    y1 = torch.sigmoid(y0)
    
    y21 = torch.mm(y1, W2torch)
    y2 = y21 + b2torch
    y3 = torch.sigmoid(y2)

    losf = nn.BCELoss()
    loss = losf(y3, torch.FloatTensor(Y))
    loss.backward()
    
    #print(abs((dy3/100-Xtorch.grad.numpy())<1e-7).all())
    print((abs(dW[0]-W1torch.grad.numpy())<1e-7).all())
    print((abs(db[0]-b1torch.grad.numpy())<1e-7).all())
    print((abs(dW[1]-W2torch.grad.numpy())<1e-7).all())
    print((abs(db[1]-b2torch.grad.numpy())<1e-7).all())



class test(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1) -> None:
        super(test, self).__init__()
        self.input_size = input_size 
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.model = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.Sigmoid(),
            nn.Linear(self.hidden_size, self.output_size)
        )
        
    def forward(self, x):
        output = self.model(x)
        return output

class test1(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1) -> None:
        super(test1, self).__init__()
        self.input_size = input_size 
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.model = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.Sigmoid(),
            nn.Linear(self.hidden_size, self.output_size),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        output = self.model(x)
        return output

def gcheck6(X, Y):
    Xtorch = torch.FloatTensor(X)
    Ytorch = torch.FloatTensor(Y)
    model_torch = test(30, 40, 1)

    y1 = model_torch.forward(Xtorch)
    lossf = nn.MSELoss()
    adam = optim.SGD(model_torch.parameters(), lr=1e-2)
    loss = lossf(y1, Ytorch)
    loss.backward()
    param = dict(model_torch.named_parameters())
    
    tgW1 = param['model.0.weight']
    tgb1 = param['model.0.bias']
    tgW2 = param['model.2.weight']
    tgb2 = param['model.2.bias']

    W1 = tgW1.detach().numpy().T
    b1 = tgb1.detach().numpy().reshape((1,-1))
    W2 = tgW2.detach().numpy().T
    b2 = tgb2.detach().numpy().reshape((1,-1))

    model = NerualNet(featureDim=30, outputDim=1, HiddenDims=[40], lastACT='linear')
    model.W = [W1, W2]
    model.b = [b1, b2]
    cahe = model.Loss(X, Y, lossfunction='MSELoss')
    dW, db, Loss = cahe['dW'], cahe['db'], cahe['loss']
    Y0 = model.predict(X)

    print((abs(Y0-y1.detach().numpy())<1e-7).all())
    print(abs(loss.detach().numpy() - Loss)<1e-7)
    print(abs(dW[0].T-tgW1.grad.numpy()).max())
    print(abs(db[0]-tgb1.grad.numpy().reshape((1,-1))).max())
    print(abs(dW[1].T-tgW2.grad.numpy()).max())
    print(abs(db[1]-tgb2.grad.numpy().reshape((1,-1))).max())

    # train
    adam.zero_grad()

    for i in range(10):
        adam.zero_grad()
        y1 = model_torch.forward(Xtorch)
        loss = lossf(y1, Ytorch)
        loss.backward()
        
        # my model
        model.trainning()
        cahe = model.Loss(X, Y, lossfunction='MSELoss')
        dW, db, Loss = cahe['dW'], cahe['db'], cahe['loss']

        param = dict(model_torch.named_parameters())
        tgW1 = param['model.0.weight']
        tgW2 = param['model.2.weight']
        print((loss.detach().numpy()-Loss).max())
        print((abs(dW[0].T-tgW1.grad.numpy())).max())
        print((abs(dW[1].T-tgW2.grad.numpy())).max())
        
        #
        adam.step()
        for i in range(model.layers):
            model.W[i] -=  dW[i]
            model.b[i] -=  db[i]
        
        print((abs(model.W[0].T-tgW1.detach().numpy())).max())
        
# check SGD with MLE : classification
def gcheck7(X, Y):
    Xtorch = torch.FloatTensor(X)
    Ytorch = torch.FloatTensor(Y)
    model_torch = test1(30, 40, 1)

    y1 = model_torch.forward(Xtorch)
    lossf = nn.BCELoss()
    adam = optim.SGD(model_torch.parameters(), lr=1e-2)
    loss = lossf(y1, Ytorch)
    loss.backward()
    param = dict(model_torch.named_parameters())
    
    tgW1 = param['model.0.weight']
    tgb1 = param['model.0.bias']
    tgW2 = param['model.2.weight']
    tgb2 = param['model.2.bias']

    W1 = tgW1.detach().numpy().T
    b1 = tgb1.detach().numpy().reshape((1,-1))
    W2 = tgW2.detach().numpy().T
    b2 = tgb2.detach().numpy().reshape((1,-1))

    model = NerualNet(featureDim=30, outputDim=1, HiddenDims=[40], lastACT='sigmod')
    model.W = [W1, W2]
    model.b = [b1, b2]
    cahe = model.Loss(X, Y, lossfunction='MLELoss')
    dW, db, Loss = cahe['dW'], cahe['db'], cahe['loss']
    Y0 = model.predict(X)

    print((abs(Y0-y1.detach().numpy())<1e-7).all())
    print(abs(loss.detach().numpy() - Loss)<1e-7)
    print(abs(dW[0].T-tgW1.grad.numpy()).max())
    print(abs(db[0]-tgb1.grad.numpy().reshape((1,-1))).max())
    print(abs(dW[1].T-tgW2.grad.numpy()).max())
    print(abs(db[1]-tgb2.grad.numpy().reshape((1,-1))).max())

    # train
    adam.zero_grad()

    for i in range(10):
        adam.zero_grad()
        y1 = model_torch.forward(Xtorch)
        loss = lossf(y1, Ytorch)
        loss.backward()
        
        # my model
        model.trainning()
        cahe = model.Loss(X, Y, lossfunction='MLELoss')
        dW, db, Loss = cahe['dW'], cahe['db'], cahe['loss']

        param = dict(model_torch.named_parameters())
        tgW1 = param['model.0.weight']
        tgW2 = param['model.2.weight']
        print((loss.detach().numpy()-Loss).max())
        print((abs(dW[0].T-tgW1.grad.numpy())).max())
        print((abs(dW[1].T-tgW2.grad.numpy())).max())
        
        #
        adam.step()
        for i in range(model.layers):
            model.W[i] -= 1e-2 * dW[i]
            model.b[i] -= 1e-2 * db[i]
        
        print((abs(model.W[0].T-tgW1.detach().numpy())).max())
    
# check Adam
def gcheck8(X, Y):
    Xtorch = torch.FloatTensor(X)
    Ytorch = torch.FloatTensor(Y)
    model_torch = test1(30, 40, 1)

    y1 = model_torch.forward(Xtorch)
    lossf = nn.BCELoss()
    adam = optim.Adam(model_torch.parameters(), lr=1e-2)
    loss = lossf(y1, Ytorch)
    loss.backward()
    param = dict(model_torch.named_parameters())
    
    tgW1 = param['model.0.weight']
    tgb1 = param['model.0.bias']
    tgW2 = param['model.2.weight']
    tgb2 = param['model.2.bias']

    W1 = tgW1.detach().numpy().T
    b1 = tgb1.detach().numpy().reshape((1,-1))
    W2 = tgW2.detach().numpy().T
    b2 = tgb2.detach().numpy().reshape((1,-1))

    model = NerualNet(featureDim=30, outputDim=1, HiddenDims=[40], lastACT='sigmod')
    model.W = [W1, W2]
    model.b = [b1, b2]
    cahe = model.Loss(X, Y, lossfunction='MLELoss')
    dW, db, Loss = cahe['dW'], cahe['db'], cahe['loss']
    Y0 = model.predict(X)

    print((abs(Y0-y1.detach().numpy())<1e-7).all())
    print(abs(loss.detach().numpy() - Loss)<1e-7)
    print(abs(dW[0].T-tgW1.grad.numpy()).max())
    print(abs(db[0]-tgb1.grad.numpy().reshape((1,-1))).max())
    print(abs(dW[1].T-tgW2.grad.numpy()).max())
    print(abs(db[1]-tgb2.grad.numpy().reshape((1,-1))).max())

    # train
    adam.zero_grad()
    ls = LrateStrategy(model.W, model.b)
    my_adam = ls.Adam()
    for i in range(10):
        adam.zero_grad()
        y1 = model_torch.forward(Xtorch)
        loss = lossf(y1, Ytorch)
        loss.backward()
        
        # my model
        model.trainning()
        cahe = model.Loss(X, Y, lossfunction='MLELoss')
        dW, db, Loss = cahe['dW'], cahe['db'], cahe['loss']

        param = dict(model_torch.named_parameters())
        tgW1 = param['model.0.weight']
        tgW2 = param['model.2.weight']
        print((loss.detach().numpy()-Loss).max())
        print((abs(dW[0].T-tgW1.grad.numpy())).max())
        print((abs(dW[1].T-tgW2.grad.numpy())).max())
        
        #
        adam.step()
        dW, db = my_adam(dW, db)
        for i in range(model.layers):
            model.W[i] -= dW[i]
            model.b[i] -= db[i]
        
        print((abs(model.W[0].T-tgW1.detach().numpy())).max())
        print((abs(model.W[1].T-tgW2.detach().numpy())).max())



# relu
def gcheck10(X, Y):
    model = NerualNet(featureDim=len(X[0]), outputDim=len(Y[0]))
    model.groupReLU = (1, 0)
    data = model.ReLU(X)
    cache = model.MSEloss(data, Y)
    dy1 = model.backwardReLUMSE(cache)
    
    Xtorch = torch.FloatTensor(X)
    Ytorch = torch.FloatTensor(Y)
    Xtorch = Variable(Xtorch, requires_grad=True)
    Ytorch = Variable(Ytorch, requires_grad=True)
    y = torch.relu(Xtorch)
    losf = nn.BCELoss()
    loss = losf(y, Ytorch)
    loss.backward(gradient=torch.ones(X.shape))
    print(abs((dy1-Xtorch.grad.numpy())<1e-7).all())

if __name__ == '__main__':
    X = np.random.randint(0, 4, (3,4))
    Y = np.random.randint(0,2, 3).reshape((-1,1))

    model = NerualNet(featureDim=10, outputDim=1)
    d = {'gamma': np.ones((1,4)), 'beta':np.ones((1,4)), 'miu':[], 'var':[]}
    res =  model.BatchNorm(X, d)
    print(res)
    #gcheck10(X, Y)