from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
# 定义函数


# 调用函数
# 这里做分词，使用空格隔开
# corpus = [
#             '我 来到 北京 清华大学',
#             '他 来到 了 中国',
#             '小明 硕士 毕业 与 中国 科学院',
#             '我 爱 北京 天安门'
#            ]
# weight,word_location,tf = TF_IDF(corpus)
# print(weight)
# print(word_location)
# print(tf)

from enum import Flag
# from locale import ERA_D_FMT
from gensim import models
from matplotlib.pyplot import flag
from nltk.corpus.reader.wordnet import WordNetError
# from nltk.grammar import FeatureGrammar
from numpy.lib.arraypad import pad
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
import gensim
from pandas.core.algorithms import mode
import sklearn
from gensim.models import Word2Vec, word2vec
import sys
import copy
from nltk.stem import SnowballStemmer

cop = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9^\x20]")  # 只保留中文、字母、数字、空格

sys.path.append("..")
sys.path.append("../code")
from util import *


def configureFile() -> dict[str, str]:
    return {
        "trainFile": "data/classification/train.csv",
        "testFile": "data/classification/test_classification.csv",
        "label": True,
    }


# ---！！！ 补充文本处理请修改此处----！！！
def processSentence(file, label=True):
    dataset = pd.read_csv(file, header=None, encoding='unicode_escape').values[1:,:]

    TrainDataset = copy.deepcopy(dataset)

    if label:
        TrainDataset0 = []
        TrainDataset1 = []
        trainSentence0 = []
        trainSentence1 = []
        valSentence = []
        vallabels = []
        loader = DataLoader(dataset)
        for traindata, trlabel, valdata, valabel in loader.KfolderData(10, shuffle=False, test=True):
            # print(traindata)
            nanid = (traindata[:, -1] == '0')
            TrainDataset0 = traindata[nanid]  # 脏数据：标签没有给出的数据
            TrainDataset1 = traindata[~nanid]  # 正常数据
            # print(TrainDataset0)

            stemmer = SnowballStemmer("english")  # 选择语言
            stop = set(stopwords.words('english'))  # 停用词表


            #
            for i in range(0, len(TrainDataset0)):
                s = re.sub(cop, ' ', TrainDataset[i][1])
                # trainSentence0.append([stemmer.stem(w) for w in s.strip().lower().split() if w not in stop])
                trainSentence0.append([w for w in s.strip().lower().split() if w not in stop])

            for i in range(0, len(TrainDataset1)):
                s = re.sub(cop, ' ', TrainDataset[i][1])
                trainSentence1.append([w for w in s.strip().lower().split() if w not in stop])

            for i in range(0, len(valdata)):
                s = re.sub(cop, ' ', valdata[i][1])
                valSentence.append([w for w in s.strip().lower().split() if w not in stop])
                if valdata[i][2]=='0':
                    vallabels.append(0)
                else:
                    vallabels.append(1)

            return trainSentence0, trainSentence1, valSentence, vallabels
    else:
        stemmer = SnowballStemmer("english")  # 选择语言
        stop = set(stopwords.words('english'))  # 停用词表
        trainSentence = []
        for i in range(0, len(TrainDataset)):
            s = re.sub(cop, ' ', TrainDataset[i][1])
            trainSentence.append([stemmer.stem(w) for w in s.strip().lower().split() if w not in stop])

        return trainSentence


def main():

    config = configureFile()
    # 读入文件，并划分label，并对句子做修剪，然后划分word
    trainSentence0, trainSentence1, valSentence, vallabels = processSentence(config["trainFile"])

    # print(valdata)
    # trainSentence0=[['Chinese', 'Beijing', 'Chinese'],
    #                 ['Chinese', 'Chinese', 'Shanghai'],
    #                 ['Chinese', 'Macao']]
    # trainSentence1 = [['Tokyo ', 'Japan ', 'Chinese']]
    # valSentence = [['Chinese', 'Chinese', 'Chinese', 'Tokyo ', 'Japan ']]
    # vallabels = [0]

    # trainSentence = processSentence(config["testFile"], label=False)
    dict0 = {}
    dict1 = {}
    all_word = []
    n0 = 0
    n1 = 0
    for row in trainSentence0:
        n0 += len(row)
        for word in row:
            if word in dict0:
                dict0[word] += 1
            else:
                dict0[word] = 1
            if word not in all_word:
                all_word.append(word)

    for row in trainSentence1:
        n1 += len(row)
        for word in row:
            if word in dict1:
                dict1[word] += 1
            else:
                dict1[word] = 1
            if word not in all_word:
                all_word.append(word)

    m = len(all_word)





    print(n0)
    print(n1)
    for x in range(99,100):
        y = x/100
        result = []
        for row in valSentence:
            p0 = 1
            p1 = 1
            for word in row:
                # print(word)
                if word in dict0:
                    p0 = p0 * (dict0[word] + m) / (n0 + m)
                    # print((dict0[word]+1)/(n0+m))
                else:
                    p0 = p0 *(1+m)/ (n0 + m)
                    # print(1/(n0+m))

                if word in dict1:
                    p1 = p1 * (dict1[word] + m) / (n1 + m)
                else:
                    p1 = p1 *(1+m) / (n1 + m)
                # print(p0,p1)
            # print(n0 / (n0 + n1))
            p0 = p0 * n0 / (n0 + n1)
            p1 = p1 * (n1 * y) / (n0 + n1)
            # print(p0, p1)
            if p0 > p1:
                result.append(0)
            else:
                result.append(1)
        acc = 0
        for i in range(len(result)):
            print(vallabels[i],result[i])
            if vallabels[i]== result[i]:
                acc += 1
        print(x, acc / len(result))






if __name__ == "__main__":
    main()