from enum import Flag
from locale import ERA_D_FMT
from gensim import models
from matplotlib.pyplot import flag
from nltk.corpus.reader.wordnet import WordNetError
from nltk.grammar import FeatureGrammar
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

def configureFile()->dict[str,str]:
    return {
        "processFile":"data/classification/test_classification.csv",
        "vectorFile":"data/classification/test_vector002.csv",
        "word2VecParam":"param/cls_word2vec002.model",
        "vector_size":100,
        "epochs":10,
        "load":True,
        "label":False,
        "regress":False
    }


def main():
    # humor regression
    config = configureFile()
    
    # 读入文件，并划分label，并对句子做修剪，然后划分word
    if config["label"]:
        textid, sentence, trainLabel= processSentence(config["processFile"], regression=config["regress"])
    else:
        textid, sentence = processSentence(config["processFile"], label=False, regression=config["regress"])

    # 保存模型文件
    if not config["load"] and config["label"]: 
        # 建立word2Vec模型
        model = Word2Vec(sentence, min_count=5, vector_size=config["vector_size"], workers=8, sg=1, window=10, batch_words=10000, epochs=config["epochs"])
        model.save(config["word2VecParam"])
    else: 
        model = Word2Vec.load(config["word2VecParam"])
    # 传入模型文件和分好词的句子，指定vector大小，将句子转化为向量
    train_vector = Sentence2Vec(model, sentence, config["vector_size"])

    # 将vector形式的句子和label拼接，并保存
    dataset = np.append(textid, train_vector, axis=1)
    if config["label"]:
        dataset = np.append(dataset, trainLabel, axis=1)
    dataset = pd.DataFrame(dataset)
    dataset.to_csv(config["vectorFile"], header=None, index=None)


#---！！！ 补充文本处理请修改此处----！！！
def processSentence(file, label=True, regression=True):
    dataset = pd.read_csv(file, header=None, encoding='unicode_escape').values 
    TrainDataset = copy.deepcopy(dataset)
    
    if regression:
        nanid = pd.isna(dataset[:,-1])
        NADataset = dataset[nanid]      # 脏数据：标签没有给出的数据
        TrainDataset = dataset[~nanid]  # 正常数据
        labels = TrainDataset[1:, -1].astype('float64').reshape((-1,1))

    else:
        labels = dataset[1:, -2].astype('float64').reshape((-1,1))

    stemmer = SnowballStemmer("english")  # 选择语言
    # get sentences
    stop = set(stopwords.words('english')) #   停用词表
    trainSentence = []
    for s in TrainDataset[1:, 1]:
        s = re.sub(cop, ' ', s)  # 删除特殊符号
        trainSentence.append([ stemmer.stem(w)  for w in s.strip().lower().split() if w not in stop])   # 去除停用词，并完成词划分，
    # 完成上一步后，trainSentence 应该就是[[word1,word2,...,wordn],...,[word1,word2,...,wordm]]的结构了

    if not label:
        return TrainDataset[1:,0].reshape((-1,1)), trainSentence
    else:
        return TrainDataset[1:,0].reshape((-1,1)), trainSentence, labels # 标签


def Sentence2Vec(model, sentence, size):
    train_vector = []
    for row in sentence:
        vec = np.zeros(size)
        cnt = 0
        for word in row:
            try:
                vec += model.wv[word]
                cnt += 1
            except:
                pass
        vec = vec/cnt if cnt != 0 else vec
        train_vector.append(vec)
    return np.array(train_vector)


if __name__ == "__main__":
    main()