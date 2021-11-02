from numpy.lib.arraypad import pad
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords 
import gensim
import sklearn
import sys
sys.path.append("..")
sys.path.append("../code")


def configureFile()->dict[str,str]:
    return {
        "trainDataFile":"data/regression/train.csv"
    }


def main():
    # humor regression
    config = configureFile()
    dataset = pd.read_csv(config["trainDataFile"], header=None, encoding='unicode_escape').values 
    nanid = pd.isna(dataset[:,-1])
    NADataset = dataset[nanid]
    TrainDataset = dataset[~nanid]
    
    # get sentences
    stop = set(stopwords('engish'))
    trainSentence = [ w for s in TrainDataset[1:, 1] for w in s.strip().split() if w not in stop]
    trainLabel = TrainDataset[1:, -1].astype('float64')

    print(dataset[:20])

if __name__ == "__main__":
    main()