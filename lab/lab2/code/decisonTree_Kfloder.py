import numpy as np
import pandas as pd
from typing import List, Mapping, Union
import operator
import json
import matplotlib.pyplot as plt 

from pandas.core.frame import DataFrame

import sys
sys.path.append('../')

def GetConfigure():
    
    return {
        'loaden' : 0,           # load the dict key for decision tree, so that it would not build the tree again 
        'loadDictFile' : 'param/decision_root_dict.json',
        'saveDictFile' : 'param/decision_root_dict.json',    
        'validFile' : 'testFile/car_test_without_label_1.csv',   # if not specify this file, use splitted default dataset for valset
        'trainFile' : 'testFile/car_train_with_label.csv',       # the same as above
        'saveResultPath':'result/18308133_liuxianbin_decisionTree_prediction',
        'validation_rate': 0.15,     # the rate of dataset spliting for validation
        'reconstruct': 0,       # whether to reconstruct the train/val dataset with last time, only in train mode
        'testMode':1,               # turn on testMode: will no longer save any param with model, last dataset. inc.
        'alg':'cart',                # id3, c4.5, cart,
        'k':0                        # use k-floder
    }
    s

def splitDataset(dataset:pd.DataFrame, Attr:Union[str, int], val:Union[int, List[int], None])-> pd.DataFrame:
    if val == None:
        val = set()
        for i in dataset[Attr]:
            val.add(i)
        val = sorted(list(val))
    
    splitData = [pd.DataFrame(columns=dataset.columns)]*(len(val))
    for i in range(len(val)):
        splitData[i] = splitData[i].append(dataset[dataset[Attr] == val[i]])

    # # TODO: for series
    # splitData = [pd.DataFrame(columns=dataset.columns)]*(len(val)+1)
    # for i in range(len(val)-1):
    #     splitData[i] = splitData[i].append(dataset[(dataset[Attr] > val[i]) & (dataset[Attr] <= val[i+1])])
    # splitData[0] = splitData[0].append(dataset[dataset[Attr] <= val[0]])
    # splitData[-1] = splitData[-1].append(dataset[dataset[Attr] > val[-1]])

    return splitData, val

def getlabel(dataset):
    labels = list(dataset.iloc[: ,-1])
    # labels = dataset['Label']
    N = len(labels)
    labelDict = dict()
    for label in labels:
        labelDict[label] = labelDict[label] + 1 if label in labelDict else 1
    return labelDict

def calHD(dataset, labelDict=None):
    # HD = - \sum_{j=1}^{l}{Dj/D * log(Dj/D)} where l is type of label
    if labelDict == None:
        labelDict = getlabel(dataset)   #get the label, and frequency of label
    N = len(dataset)
    HD = 0
    for label, freq in labelDict.items():
        HD += -freq/N * np.log(freq/N)
    return HD

def ID3(dataset, attr, HD):
    HD_A = 0
    N = len(dataset)
    spData, _ = splitDataset(dataset, attr, None)
    for each in spData:
        prob = len(each) / N
        # HD_A = - \sum_{j=1}^{v}{Dj/D * HD(Dj)}
        HD_A += prob * calHD(each)
    return HD - HD_A


def GainRatio(dataset, attr, HD):
    HD_A = 0
    N = len(dataset)
    assert N != 0
    spData, _ = splitDataset(dataset, attr, None)
    Dj_Div_D = np.array([len(each) for each in spData]).reshape(1, -1) / N

    # splitInfo_with_a = - \sum_{j=1}^{v} { Dj/D * log(Dj/D) }, where v is the values of attribute a, Dj is the set with attr of v
    spInfo = max(- Dj_Div_D.dot(np.log(Dj_Div_D.T)), 1e-3)

    assert spInfo != 0

    for each in spData:
        prob = len(each) / N
        HD_A += prob * calHD(each)
    return (HD - HD_A) / spInfo


def Cart(dataset, attr):
    # gini = \sum_{j=1}^{v} {Dj/D * gini(Dj)}

    def Gini(data):
        # gini = 1 - \sum_{j=1}^{l} {(Dj/D)^2}, where l is type of label
        GD = 0
        N = len(data)
        labelDict = getlabel(data)
        for key, val in labelDict.items():
            GD += np.square(val / N)
        # 1 - sum(prob^2)
        return 1-GD

    Cart = 0
    N = len(dataset)
    spData, _ = splitDataset(dataset, attr, None)
    
    for each in spData:
        prob = len(each) / N
        Cart += prob * Gini(each)
    return Cart

# how to represent the tree ? 
class decisionTress():
    def __init__(self, dataset) -> None:
        # attr : (L:means label)
        # dec: decision

        self.root = {'attr':'', 'dec':{}, 'pred':0}
        self.dataset = dataset


    def bulid(self, alg:str='ID3') -> None:
        def choose(root : dict, dataset:pd.DataFrame):
            # leaf
            labelDict = getlabel(dataset)
            thisLeafPred = max(labelDict.items(), key=operator.itemgetter(1))[0]
            root['pred'] = thisLeafPred
            if len(labelDict) == 1:         # when the label of dataset is the same
                root['attr'] = 'L'
                root['dec'] = thisLeafPred
                root['pred'] = thisLeafPred
                return
            elif len(dataset.columns) == 1: # or when all attribution has been ran out
                root['attr'] = 'L'
                root['dec'] = thisLeafPred
                root['pred'] = thisLeafPred
                return

            # choose feature}
            bestcol, bestg = '', -np.inf
            HD = calHD(dataset, labelDict)
            for col in dataset.columns[:-1]:
                if alg.lower() == 'cart':
                    g = -Cart(dataset, col)
                elif alg.lower() == 'c4.5':
                    g = GainRatio(dataset, col, HD)
                else:
                    g = ID3(dataset, col, HD)

                if bestg < g:
                    bestcol, bestg = col, g
            
            # split dataset
            spData, val = splitDataset(dataset, bestcol, None)
            root['attr'] = bestcol
            # recursion
            for i in range(len(spData)):
                del spData[i][bestcol]
                root['dec'][val[i]] = {'attr':'', 'dec':{}, 'pred':0}
                choose(root['dec'][val[i]], spData[i])
        
        choose(self.root, self.dataset)

    def predictSingle(self, data):
        cur = self.root
        while cur['attr'] != 'L':
            col = cur['attr']
            if data[col] not in cur['dec']:
                return cur['pred']
            else:
                cur = cur['dec'][data[col]]
        return cur['dec']   


def draw(res, label):
    if label == 'id3':
        plt.plot(res[0],res[1],'r--',label=label)
    if label == 'c4.5':
        plt.plot(res[0],res[1],'g--',label=label)
    if label == 'cart':
        plt.plot(res[0],res[1],'b--',label=label)



def test2():    
    # deal with k floder
    config = GetConfigure()
    dataset = pd.read_csv('lab2_dataset/car_train.csv').sample(frac=1).reset_index(drop=True)

    # use k-floder cross validation split the dataset
    ks = np.linspace(2, 19, 3).astype('int')
    for alg in ['id3', 'c4.5', 'cart']:
        total_ac = []
        for k in ks:
            tmp_ac = 0
            num = int(len(dataset) / k)
            for i in range(1, k+1):
                ac = 0
                # choose the i-th part for validation   
                train_set = dataset.iloc[:num*(i-1)].append(dataset.iloc[num*i:]).reset_index(drop=True)
                validation_set = dataset.iloc[num*(i-1):num*(i)].reset_index(drop=True)
                t = decisionTress(train_set)
                t.bulid(alg=alg)
                for i in range(len(validation_set)):
                    label = t.predictSingle(validation_set.iloc[i])
                    ac += (label == validation_set.iloc[i]['Label'])
                ac /= len(validation_set)
                tmp_ac += ac
            tmp_ac /= k
            total_ac.append(tmp_ac)
        draw((ks, total_ac), alg)
    plt.title('result of every k and every algorithm')
    plt.xlabel('k')
    plt.ylabel('accuarcy')
    plt.legend()
    plt.show()

def test1():
    # dataset = [[1,1,1],[0,1,0],[1,0,0],[0,0,0]]
    config = GetConfigure()
    dataset = pd.read_csv('lab2_dataset/car_train.csv')
    
    # if not reconstruct with last data, use another random data for constructing the train/val set
    if config['validFile'] == '' or config['trainFile'] == '':
        if not config['reconstruct']:
            dataset = dataset.sample(frac=1)
            # split datas
            N = len(dataset)
            nums = int((1-config['validation_rate'])*N)
            index_train = dataset.index[:nums]
            index_val = dataset.index[nums:]
            if not config['testMode']:
                # save the dataset loaded this time if testMode is turn off
                pd.DataFrame(data=index_train,columns=['train']).to_csv('lastTrainIndex.csv', index=False)
                pd.DataFrame(data=index_val,columns=['val']).to_csv('lastValIndex.csv', index=False)

        else:   # reconstruct:  load from last read
            indexes = pd.read_csv('param/lastTrainIndex.csv', index_col=None)
            index_train = list(indexes['train'])
            indexes = pd.read_csv('param/lastValIndex.csv')
            index_val = list(indexes['val'])
    
    if config['validFile'] == '':
        # get validation set with default dataset
        validation_set = dataset.iloc[index_val].reset_index(drop=True)
    else:
        validation_set = pd.read_csv(config['validFile'])

    if config['trainFile'] == '':
        # get train set with default dataset
        train_set = dataset.iloc[index_train].reset_index(drop=True)
    else:
        train_set = pd.read_csv(config['trainFile'])


    Tree = decisionTress(train_set)

    if config['loaden']:
        with open(config['loadDictFile'], 'r') as f:
            Tree.root = json.load(f)
    else:
        # not load last key, recalculate
        Tree.bulid(config['alg'])
        if not config['testMode']:
            with open(config['saveDictFile'], 'w') as f:
                json.dump(Tree.root, f)

    # predict
    labeled = not validation_set['Label'].isnull().all()
    labels = pd.DataFrame(columns=list(validation_set.columns)+['predlabel'])
    for i in range(len(validation_set)):
        label = Tree.predictSingle(validation_set.iloc[i])
        labels = labels.append(dict(zip(labels.columns, list(validation_set.iloc[i])+[label])), ignore_index=True)
    ac = ''
    if labeled:
        ac = '_%1.5f' % (labels['Label'] == labels['predlabel']).mean()
        tr = 'default_split_file' if config['trainFile'] == '' else config['trainFile']
        val = 'default_split_file' if config['validFile'] == '' else config['validFile']
        if not config['reconstruct']:
            print('the ac rate for train_set: %s on validation_set:%s is %s' % (tr, val, ac))
        else:
            print('the ac rate on validation_set:%s with last key is %s' % (val, ac))
            
    if config['testMode']:
        labels.to_csv('testFile/'+config['saveResultPath']+'_vrate_%s_'%config['validation_rate']+config['alg']+ac+'.csv')
    else:
        labels.to_csv(config['saveResultPath']+'_vrate_%s_'%config['validation_rate']+config['alg']+ac+'.csv')
    # print(Tree.predictSingle([1,0]))


if __name__ == '__main__':
    if (GetConfigure()['k'] == 0):
        test1()
    else:
        test2()