import sys
from numpy.lib.function_base import disp
import pandas
from numpy import float64, ma, maximum
import numpy as np 

sys.path.append('../code')
from knn import KNN
from tfidf import encoder_tf
from classification import readtestfile, readfile, to_label, draw


class KNNRegression(KNN):
    def __init__(self, k=1, dtype=float64) -> None:
        super().__init__(k=k, dtype=dtype)

        # avoid repeatly computing the distance
        self.predict_data_loaded = np.array([])
        self.distance_loaded = {}

    # here Y comes with the prob of every label
    def train(self, X, Y):
        # train the KNN module with data X and label y
        self.X = np.array(X)
        self.y = np.array(Y).astype(float64)

    def distance(self, X, distType=2, epsilon=1e-5):
        dist = super().distance(X, distType=distType)
        return np.maximum(dist, epsilon)
    
    def predict(self, X, k=1, lossType=2):
        self.K = k
        Y = []
        if (self.predict_data_loaded.shape != X.shape) or (self.predict_data_loaded != X).any() or lossType not in self.distance_loaded:      # any data loaded inequal to X : X is a new one
            # loss type is the same as distType
            # predict the label of input vector x
            self.predict_data_loaded = X
            dist = self.distance(X, lossType)       # dist_k contain index of the least K distance
            self.distance_loaded[lossType] = dist
        else:
            dist = self.distance_loaded[lossType]
        dist_K = np.argsort(dist)[:,:self.K]  # the nearest K smaples
        # Y = np.argmax(dist_K, axis=1)
        Y_K = self.y[dist_K]        # shape : <N, K, C>
        for i in range(len(Y_K)):
            dist_k_i = dist[i][dist_K[i]].reshape(self.K,1)    # get the K distances
            # note that np.sum should point out the sum for the columns,
            # and the Denominator : " 1/dist_k_i" used to make the sum of prob to be 1
            Y.append(np.sum(Y_K[i] / dist_k_i, axis=0) / np.sum(1/dist_k_i)) # of shape <K, C> 
               # Y_K[i][k] / distance_k
        
        return np.array(Y)


def cor(X, Y):
    # X: N, D  Y:N,D
    assert X.shape == Y.shape
    X_mean = np.mean(X, axis=0)
    Y_mean = np.mean(Y, axis=0)
    X_norm = X-X_mean
    Y_norm = Y-Y_mean
    
    return np.mean(np.sum(X_norm * Y_norm, axis=0) / np.sqrt(np.sum(X_norm**2, axis=0)*np.sum(Y_norm**2, axis=0)))


def main():
    configfile = {
        "data_file_folder": '/Users/liuxb/2021_au/al_rao/lab/lab1/lab1_data/regression_dataset',
        "check_file_folder":'/Users/liuxb/2021_au/al_rao/lab/lab1/for_checking',
        "check_test_file":  'regression_simple_test.csv',
        "test_file":        'test_set.csv',
        "save_path":        '/Users/liuxb/2021_au/al_rao/lab/lab1/result/18308133_liuxianbin_regression.csv',
        "check_mode":       0,
        "output_validation":1
    }


    folder = configfile['data_file_folder']
    savepath = configfile['save_path']

    train_file = 'train_set.csv'
    validation_file = 'validation_set.csv'
    check_file = configfile['check_test_file']
    test_file = configfile['test_file']
    # test_file = 'test_set.csv'
    label_map = {'anger':0, 'disgust':1, 'fear':2, 'joy':3, 'sad':4, 'surprise':5}
    train_set, label_tr, vocab, label_map, idf = readfile(folder+'/'+train_file, label_map=label_map, eval=False, label_type=2)
    val_set, label_v, _, _, _= readfile(folder+'/'+validation_file, vocab, label_map, True,  idf_vector=idf, label_type=2)
    if configfile['check_mode']:
       test_set, content = readtestfile(configfile['check_file_folder']+'/'+check_file, vocab, idf, check_mode=configfile['check_mode'])
    else:
       test_set, content = readtestfile(folder+'/'+test_file, vocab, idf, check_mode=configfile['check_mode'])

    
    model = KNNRegression()
    # ks = [1,2,3,5,10,15,20]
    ks = np.linspace(5, 10, 6).astype('int')
    L = [1,2,3]
    ac = {}
    best_ac = -1
    best_config = ()

    for k in ks:
        for losstype in L:
            model.train(train_set, label_tr)
            y_pred = model.predict(val_set, k=k, lossType=losstype)
            
            thisac = cor(y_pred, label_v.astype(float64))

            print("the cor of config(k:%d, losstype:%d) is %f" %(k, losstype, thisac))
            if thisac > best_ac:
                best_ac = thisac
                best_config = (k, losstype)
            
            ac[(k, losstype)] = thisac

    # save best_validation
    y_pred = model.predict(val_set, best_config[0], best_config[1])
    with open(file="/Users/liuxb/2021_au/al_rao/lab/lab1/result/validation_predict.csv", mode='wb') as f:
        df = pandas.DataFrame()
        df['textid'] = list(range(1, len(y_pred)+1))
        i = 0
        for name in label_map:
            df[name] = y_pred[:,i]
            i += 1

        df.to_csv(f, index=False)
        print('already save the best validation result')

    # draw
    #draw(ac)
    
    # go for test 
    # y_pred = model.predict(test_set, best_config[0], best_config[1])
    print('best config for validation set is: k:%d, losstype:%d, best cor:%f'%(best_config[0], best_config[1], best_ac))
    y_pred = model.predict(test_set, best_config[0], best_config[1])
    # save in csv form
    

    # label of y_test

    with open(file=savepath, mode='wb') as f:
        df = pandas.DataFrame()
        df['textid'] = list(range(1, len(content)+1))
        df['Words (split by space)'] = content
        i = 0
        for name in label_map:
            df[name] = y_pred[:,i]
            i += 1

        df.to_csv(f, index=False)
        print('already save the test result into file:%s'%(savepath))



if __name__ == '__main__':
    main()


def test_cor():
    # a = np.array([[1,3],[2, 4]])
    # b = np.array([[2,3],[4,7]])
    a = np.array([[1,2],[2,3],[5,4]])
    b =np.array([[1,2],[2,4], [3,4]])
    print(cor(a,b))


