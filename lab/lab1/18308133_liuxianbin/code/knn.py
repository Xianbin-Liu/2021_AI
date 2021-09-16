from numpy.linalg import norm
from pandas.core.algorithms import SelectNFrame, mode
import sys
import numpy as np
from numpy.core.fromnumeric import argmax
from numpy.lib.function_base import disp
from numpy.lib.index_tricks import AxisConcatenator  

class KNN(object):
    def __init__(self, k=1, dtype=np.float32) -> None:
        self.K = k
        pass

    def train(self, X, y):
        # train the KNN module with data X and label y
        self.X = X
        self.y = y
        self.X = np.array(X)
        self.y = np.array(y).astype(int)

    def distance(self, X, distType=2):
        # calculate the distance with different distance type
        # dist_type == 1 : Manhoton distance ; while 2 means Euler distance
        #              3 means cos distance 
        # the distance of target vector X_target with the other vector saved when calling train
        X.astype('float32')
        
        # if distType == 1:
        #     return np.sqrt(np.sum(self.X - x, axis=1))
        # elif distType == 2:
        #     return np.sqrt(np.sum(np.square(self.X - x), axis=1))
        
        # for faster, now calculate in vector form for parallel computing 
        # (X-Y)^2 = X^2 - 2*X*Y + Y^2  in file: 
        if distType == 1:
            res = []
            for i in range(len(X)):
                res.append(np.sum(np.abs(self.X-X[i]), axis=1))
            return np.array(res)

        elif distType == 2:
            res = np.sum(X**2, axis=1, keepdims=1) + np.sum(self.X**2, axis=1) - 2.0*X.dot(self.X.T)
            res = np.maximum(res, 0)            # because the float computing could make the res smaller than 0 when its actual result is close to zero
                                                # or the error: RuntimeWarning: invalid value encountered in sqrt occurred
            return np.sqrt(res)

        else:       # cosines = A.*B / |A||B|
            # but it's not a distance while the value of consine bigger the distance smaller
            # res = []
            # mod_train_X = np.linalg.norm(self.X, axis=1)
            # for i in range(len(X)):
            #     norm_Xi = np.linalg.norm(X[i])
            #     res.append(X[i].dot(self.X.T)/(norm_Xi*mod_train_X))
            # return 1-np.array(res)
            
            # in vector way
            self_X_norm = self.X / np.linalg.norm(self.X, axis=1, keepdims=1)
            X_norm = X / np.linalg.norm(X, axis=1, keepdims=1)
            return 1 - X_norm.dot(self_X_norm.T)

    def predict(self, X, k=2, lossType=2):
        self.K = k
        Y = np.zeros(len(X))
        # loss type is the same as distType
        # predict the label of input vector x
        dist = self.distance(X, lossType)
        dist_K = np.argsort(dist)[:,:self.K]
        # Y = np.argmax(dist_K, axis=1)
        Y_K = self.y[dist_K]
        for i in range(len(Y_K)):
            # get the label of largest number in the K labels
            # use bincount for counting the numbers
            maxp = np.bincount(Y_K[i])
            Y[i] = argmax(maxp)
        Y = Y.astype(int).reshape(-1)
        return self.y[Y]
    
# X = np.array([[1,2,3],[4,5,6]])
# y = np.array([1,1,1])
# X_train = np.array([[1,1,1],[1,2,3],[2,2,2]])

# ml = KNN()
# ml.train(X_train, y)
# print(ml.distance(X))

def main():
    model = KNN()
    train_X = np.array([[1,2,3],[2,3,4]])
    train_y = np.array([1,2])
    model.train(train_X, train_y)
    print(model.distance(np.array([[2,3,4],[2,3,1]]), 1))

if __name__ == '__main__':
    main()
