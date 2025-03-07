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
from sklearn.svm import LinearSVC
import error

def getFeature(dir, years):
    MalwareCorpus=os.path.join(dir, "training", "malware")
    GoodwareCorpus=os.path.join(dir, "training", "goodware")
    AllMalSamples = CM.ListFiles(MalwareCorpus, ".data", year=years)
    AllGoodSamples = CM.ListFiles(GoodwareCorpus, ".data", year=years)
    return AllMalSamples, AllGoodSamples

def main():
    dir = "/home/wang/Data/android"
    train_years = [str(i) for i in range(2014, 2020)]
    test_years = [str(i) for i in range(2020, 2024)]
    malwares, goodwares = getFeature(dir, train_years)
    tmalwares, tgoodwares = getFeature(dir, test_years)
    NewFeatureVectorizer = TF(input='filename', tokenizer=lambda x: x.split('\n'), token_pattern=None,binary=True)
    NewFeatureVectorizer.fit(malwares+goodwares+tmalwares+tgoodwares)
    goodfeatures = NewFeatureVectorizer.transform(goodwares)
    malfeatures = NewFeatureVectorizer.transform(malwares)
    tgoodfeatures = NewFeatureVectorizer.transform(tgoodwares)
    tmalfeatures = NewFeatureVectorizer.transform(tmalwares)


    train_features = vstack([goodfeatures, malfeatures])
    test_features = vstack([tgoodfeatures, tmalfeatures])
    train_labels = np.concatenate([np.zeros(goodfeatures.shape[0]), np.ones(malfeatures.shape[0])])
    test_labels = np.concatenate([np.zeros(tgoodfeatures.shape[0]), np.ones(tmalfeatures.shape[0])])
    train_features, train_labels = shuffle(train_features, train_labels, random_state=1423)
    test_features, test_labels = shuffle(test_features, test_labels, random_state=1423)
    svcModel = LinearSVC(max_iter=1000000, C=1, dual=False, fit_intercept=False)
    svcModel.fit(train_features, train_labels)

    # train_preds = svcModel.predict(train_features)
    # test_preds = svcModel.predict(test_features)
    test_f1, train_f1, acc, train_acc, test_loss, train_loss = error.evaluation_metrics(f"Without Support shift", svcModel, test_features, train_features, test_labels, train_labels)
    

    
    
if __name__=="__main__":
    main()
