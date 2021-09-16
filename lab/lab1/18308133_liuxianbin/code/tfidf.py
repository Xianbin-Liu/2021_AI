import numpy as np
import string

from numpy.core.fromnumeric import searchsorted
from numpy.lib.type_check import common_type

def encoder_tf(content, vocab, epsilon=1e-4):
    tf = np.zeros((len(content), len(vocab)))
    for i in range(len(content)):
        for word in content[i]:
            if word in vocab:
                tf[i][vocab[word]] += 1
    tf += epsilon
    tf /= np.sum(tf, axis=1, keepdims=1)
    return tf

def output(filename):
    with open(file=filename, mode='r') as f:
        lines = f.readlines()
        content = []
        idf = {}
        vocab = {}
        for line in lines:
            line = line.split('\t')[-1].strip('\n').split(' ')
            exist = set()
            for word in line:
                if word not in vocab:
                    vocab[word] = len(vocab)
                if word not in exist:   # word在这个句子还没出现过
                    index = vocab[word]       # 取出word对应下标
                    idf[index] = idf[index] + 1 if index in idf else 1  # 技术+1
                    exist.add(word)
            content.append(line)
        idf_vector = []
        for _, value in idf.items():
            idf_vector.append(value)
        idf_vector = np.array(idf_vector)
        idf_vector = np.log(len(content) / (1+idf_vector))
        tf = encoder_tf(content, vocab, epsilon=0)
        tfidf_matrix =tf*idf_vector
        # print(tfidf_matrix)
        return tfidf_matrix


if __name__ == '__main__':
        floder = '/Users/liuxb/2021_au/al_rao/lab/lab1'
        # np.savetxt(floder+'/result/18308133_liuxianbin_TFIDF.csv', output(floder+'/lab1_data/tfidf_dataset/semeval.txt'), delimiter=',')
        # np.savetxt(floder+'/result/18308133_liuxianbin_TFIDF_sample.csv', output(floder+'/lab1_data/tfidf_dataset/semeval_sample.txt'), delimiter=',')
        m = output(floder+'/lab1_data/tfidf_dataset/semeval_sample.txt')
        with open(floder+'/result/18308133_liuxianbin_TFIDF_sample.txt', 'w') as f:
            for line in m:
                for num in line:
                    if num != 0:
                        f.write("%6.10f\t"% (num))
                f.write('\n')

        m = output(floder+'/lab1_data/tfidf_dataset/semeval.txt')
        with open(floder+'/result/18308133_liuxianbin_TFIDF.txt', 'w') as f:
            for line in m:
                for num in line:
                    if num != 0:
                        f.write("%6.10f\t" % (num))
                f.write('\n')
        