import numpy as np
import pandas
import csv
# from ..code.tfidf import *
# from ..code.knn import *
import sys
sys.path.append('../code')
from knn import *
from tfidf import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def readtestfile(filename, vocab, idf_vector, check_mode):
    with open(filename, 'r') as f:
        content = f.readlines()[1:]
        # for test
        if check_mode:
            content = [line.strip('\n').split(',')[0] for line in content]
        else:
            content = [line.split(',')[1] for line in content[1:]]
        sentence = [line.split(' ') for line in content]
        td = encoder_tf(sentence, vocab)
        return td*idf_vector, content

def readfile(filename, prevocab={}, label_map={}, eval=False, idf_vector=[], label_type=1):
    ''' transform the string-like content from $<filename> into tf-idf form:
        $ labeled:  bool, indicate whether the label exist in the file
        $ prevocab: vocabulary <dict>, in form: <word>:<index(in one hot vector)>
        $ label_map:the <dict> of label, in form <string>:<id>     i.e. 'joy':0
        $ eval:     bool, evaluation mode
    '''
    with open(file=filename, mode='r') as f:
        lines = f.readlines()[1:]
        # lines = f.readlines()
        content = []            # record the split word in line
        labels = []             # record the label(if existed) in line 
        label_map = label_map   # load the label map
        vocab = prevocab        # load the vocabulary library
        idf = {}                # idf of <word_j>:<times occured in different content>
        idf_vector = idf_vector
        for i in range(len(vocab)):
            idf[i] = 0
        
        for line in lines:
            if label_type == 1:
                sentence, label = line.strip('\n').split(',')
                # sentence= line.split('\t')[-1].strip('\n') # for test
                # label = []
            else:
                line = line.strip('\n').split(',')
                sentence = line[0]
                labels.append(line[1:])        # here label means the prob of each emotion

            sentence = sentence.split(' ')
            exist = set()

            # update idf and vocab if in trainning file
             # if eval mode on, no extend the vocabulary, use the vocabulary loaded
             # and idf calc in only trainning mode
            if not eval:
                for word in sentence:
                    if word not in vocab:
                        vocab[word] = len(vocab)
                    if word not in exist:
                        index = vocab[word]
                        idf[index] = idf[index] + 1 if index in idf else 1
                        exist.add(word)
            if label_type == 1:
                # update label map
                if label not in label_map:
                        label_map[label] = len(label_map)
                
                # if labeled:
                labels.append(label_map[label])

            content.append(sentence)
            
        # print(content)
        # encoder to tf matrix 
        tf = encoder_tf(content, vocab)     # tf_matrix
        # print(tf)
        # idf_matrix = log( D / (1+times) ), times is just the var idf here
        if not eval:
            for _, value in idf.items():        # convert dict to vector form
                idf_vector.append(value)
            idf_vector = np.array(idf_vector)
            idf_vector = np.log(len(content) / (1+idf_vector))
        # print(idf_vector)
        tfidf_matrix = tf*idf_vector
        
        if label_type == 2:
            labels = np.array(labels)

    return tfidf_matrix, labels, vocab, label_map, idf_vector # for output test_set result

def draw(ac):
    ks, losstypes, accs = [],[],[]
    for item in ac.items():
        (k, losstype), acc = item
        ks.append(k)
        losstypes.append(losstype)
        accs.append(acc)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(ks, losstypes, accs)
    ax.set_xlabel('k')
    ax.set_ylabel('losstype')
    ax.set_zlabel('accuracy')
    plt.show()

def to_label(numbers, label_map):
    # according the label_map, convert the label from 'int'(start from 0) to 'string'
    reversed_label = {}
    result = []
    for key, value in label_map.items():
        reversed_label[value] = key
    for num in numbers:
        result.append(reversed_label[num])
    return result



def main():
    configfile = {
        "data_file_folder": '/Users/liuxb/2021_au/al_rao/lab/lab1/lab1_data/classification_dataset',
        "check_file_folder":'/Users/liuxb/2021_au/al_rao/lab/lab1/for_checking',
        "check_test_file":  'classification_simple_test.csv',
        "test_file":        'test_set.csv',
        "save_path":        '/Users/liuxb/2021_au/al_rao/lab/lab1/result/checking_classification.csv',
        "check_mode":       1
    }


    # folder_ts = '/Users/liuxb/2021_au/al_rao/lab/lab1/lab1_data/tfidf_dataset'
    # ts_set,_,_,__,_ = readfile(folder_ts+'/'+'semeval_sample.txt', eval=False)

    folder = configfile['data_file_folder']
    train_file = "train_set.csv"
    validation_file = 'validation_set.csv'
    test_file = configfile['test_file']
    check_test_file = configfile['check_test_file']
    train_set, label_tr, vocab, label_map, idf = readfile(folder+'/'+train_file)
    val_set, label_v, _, label_map, _= readfile(folder+'/'+validation_file, vocab, label_map, eval=True,  idf_vector=idf)
    if configfile['check_mode']:
        test_set, content = readtestfile(configfile['check_file_folder']+'/'+check_test_file, vocab, idf, check_mode=configfile['check_mode'])
    else:
        test_set, content = readtestfile(configfile['data_file_folder']+'/'+test_file, vocab, idf, check_mode=configfile['check_mode'])
        
#    ks = [1,2,3,5,10,15,20]
    ks = np.linspace(6, 12, 6).astype('int')
    L = [1,2,3]
    ac = {}
    best_ac = -1
    best_config = ()

    model = KNN()
    for k in ks:
        for losstype in L:
            model.train(train_set, label_tr)        # save the training data
            y_pred = model.predict(val_set,k=k, lossType=losstype)  # predict
            thisac = np.mean(y_pred == label_v)     # calculate the accuracy
            print("the accuracy of config(k:%d, losstype:%d) is %f" %(k, losstype, thisac))
            if thisac > best_ac:
                best_ac = thisac                    # update the best performance
                best_config = (k, losstype)
            ac[(k, losstype)] = thisac              # record the evaluation for the config

    # draw
    #draw(ac)

    print("\n the best config is k(%d), losstype(%d) with accuracy %f"%(best_config[0], best_config[1], best_ac))

    # go for test 
    y_pred = model.predict(test_set, best_config[0], best_config[1])
    y_pred.reshape((len(y_pred)))
    y_pred_label = to_label(y_pred, label_map)
    
    # save in csv form


    with open(file=configfile['save_path'], mode='wb') as f:
        df = pandas.DataFrame()
        df['textid'] = list(range(1, len(content)+1))
        df['Words (split by space)'] = content
        df['label'] = y_pred_label
        df.to_csv(f, index=False)

if __name__ == '__main__':
    # for test td-idf:
    main()




def test():
    folder = '/Users/liuxb/2021_au/al_rao/lab/lab1/lab1_data/classification_dataset'
    train_file = 'train_set.csv'
    train_set, label_tr, vocab, label_map = readfile(folder+'/'+train_file, labeled=True)

    validation_file = 'validation_set.csv'
    valid_set, label_v, _ , _ = readfile(folder+'/'+validation_file, prevocab=vocab,labeled=True, eval=True)

    # expand the vocab with the new vocab set
    train_set = np.hstack((train_set, np.zeros((len(train_set), len(vocab)-len(train_set[0])))))

    test_set = readtestfile()

    model = KNN()
    model.train(train_set, label_tr)
    y_pred = model.predict(valid_set, k=1)
    ac = np.mean(y_pred == label_v)
    print("ac:%f" %(ac))
    # print(to_label(numbers=y_pred, label_map=label_map))
    res = to_label(numbers=y_pred, label_map=label_map)
    # load the test set
    with open(file=folder+'/'+'test_set.csv', mode='r') as f:
        pass

    with open(file='/Users/liuxb/2021_au/al_rao/lab/lab1/result/testing_result.txt', mode='wb') as f:
        writer =csv.writer(f)
        df = pandas.DataFrame(res)
        df.to_csv(f, header=False)