import torch
import torch.nn as nn
from itertools import cycle
import os
import CommonModules as CM
from sklearn.feature_extraction.text import TfidfVectorizer as TF
import numpy as np
from scipy.sparse import vstack
import pickle
from sklearn.utils import shuffle
import gc
from sklearn.svm import LinearSVC, SVC
import error
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
import random

def getFeature(dir, years):
    MalwareCorpus=os.path.join(dir, "training", "malware")
    GoodwareCorpus=os.path.join(dir, "training", "goodware")
    AllMalSamples = CM.ListFiles(MalwareCorpus, ".data", year=years)
    AllGoodSamples = CM.ListFiles(GoodwareCorpus, ".data", year=years)
    return AllMalSamples, AllGoodSamples


def main():
    dir = "/home/wang/Data/android"
    device = torch.device("cpu")
    year1 = [str(i) for i in range(2014, 2020)]
    year2 = [str(i) for i in range(2020, 2024)]
    malwares1, goodwares1 = getFeature(dir, year1)
    malwares2, goodwares2 = getFeature(dir, year2)
    NewFeatureVectorizer = TF(input='filename', tokenizer=lambda x: x.split('\n'), token_pattern=None,binary=True)
    NewFeatureVectorizer.fit(malwares1+goodwares1+malwares2+goodwares2)
    labels1 = np.hstack([np.zeros(len(goodwares1)), np.ones(len(malwares1))])
    labels2 = np.hstack([np.zeros(len(goodwares2)), np.ones(len(malwares2))])
    train1_x, test1_x, train1_y, test1_y = train_test_split(goodwares1+malwares1, labels1, test_size=int(0.2*(len(goodwares1)+len(malwares1))),
                                                     random_state=random.randint(0, 100), stratify=labels1)
    train2_x, test2_x, train2_y, test2_y = train_test_split(goodwares2+malwares2, labels2, test_size=int(0.2*(len(goodwares2)+len(malwares2))),
                                                     random_state=random.randint(0, 100), stratify=labels2)
    
    train_features1 = NewFeatureVectorizer.transform(train1_x)
    test_features1 = NewFeatureVectorizer.transform(test1_x)
    train_features1, train_labels1 = shuffle(train_features1, train1_y, random_state=2314)
    test_features1, test_labels1 = shuffle(test_features1, test1_y, random_state=2314)
    svcModel = SVC(kernel='rbf', C=1, gamma='scale')
    svcModel.fit(train_features1, train_labels1)
    _,_,_,_,_,_ = error.evaluation_metrics(f"set 1 only: ", svcModel, test_features1, train_features1, test_labels1, train_labels1)
    
    
    train_features2 = NewFeatureVectorizer.transform(train2_x)
    test_features2 = NewFeatureVectorizer.transform(test2_x)
    train_features2, train_labels2 = shuffle(train_features2, train2_y, random_state=2314)
    test_features2, test_labels2 = shuffle(test_features2, test2_y, random_state=2314)
    svcModel = SVC(kernel='rbf', C=1, gamma='scale')
    svcModel.fit(train_features2, train_labels2)
    _,_,_,_,_,_ = error.evaluation_metrics(f"set 2 only: ", svcModel, test_features2, train_features2, test_labels2, train_labels2)
    
    
    train_features = vstack([train_features1, train_features2])
    test_features = vstack([test_features1, test_features2])
    train_labels = np.hstack([train_labels1, train_labels2])
    test_labels = np.hstack([test_labels1, test_labels2])
    train_features, train_labels = shuffle(train_features, train_labels, random_state=2314)
    test_features, test_labels = shuffle(test_features, test_labels, random_state=2314)
    svcModel = SVC(kernel='rbf', C=1, gamma='scale')
    svcModel.fit(train_features, train_labels)
    _,_,_,_,_,_ = error.evaluation_metrics(f"set 1 + set 2 tested on set 1 + set2: ", svcModel, test_features, train_features, test_labels, train_labels)
    _,_,_,_,_,_ = error.evaluation_metrics(f"set 1 + set 2 tested on set 1: ", svcModel, test_features1, train_features1, test_labels1, train_labels1)
    _,_,_,_,_,_ = error.evaluation_metrics(f"set 1 + set 2 tested on set 2: ", svcModel, test_features2, train_features2, test_labels2, train_labels2)
    
    
    
    


if __name__=="__main__":
    main()