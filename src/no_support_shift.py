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
from sklearn.metrics import accuracy_score

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
    
    
    iterations = 100
    for i in range(10, iterations, 10):
        print(f"Iteration: {i}")
        test_scores = svcModel.decision_function(test_features)
        best_thresh = np.percentile(np.abs(test_scores), i)  
        print(f"Threshold for top {100-i}% confident samples: {best_thresh}")
    
        high_confidence_idx = np.where(np.abs(test_scores) >= best_thresh)[0]
        # pseudo_labels = (test_scores[high_confidence_idx] >= 0).astype(int)
    
        pseudo_labels = test_labels[high_confidence_idx]
        pseudo_features = test_features[high_confidence_idx]
        
        num_positive = np.sum(pseudo_labels)  
        num_negative = len(pseudo_labels) - num_positive
        total_selected = len(pseudo_labels)

        print(f"Total selected samples: {total_selected}")
        print(f"Positive class (1): {num_positive} ({num_positive / total_selected:.2%})")
        print(f"Negative class (0): {num_negative} ({num_negative / total_selected:.2%})")
        
        # train_features = vstack([train_features, pseudo_features])
        # train_labels = np.concatenate([train_labels, pseudo_labels])

        # train_features, train_labels = shuffle(train_features, train_labels, random_state=1423)
    
        # svcModel = LinearSVC(max_iter=1000000, C=1, dual=False, fit_intercept=False)
        # svcModel.fit(train_features, train_labels)
    
        test_f1, train_f1, acc, train_acc, test_loss, train_loss = error.evaluation_metrics(f"Naive Iteration {i}", svcModel, pseudo_features, train_features, pseudo_labels, train_labels)
        # '''
    for i in range(91, 101, 1):
        print(f"Minor Iteration: {i}")
        test_scores = svcModel.decision_function(test_features)
        best_thresh = np.percentile(np.abs(test_scores), i)  
        print(f"Threshold for top {100-i}% confident samples: {best_thresh}")
    
        high_confidence_idx = np.where(np.abs(test_scores) >= best_thresh)[0]
        # pseudo_labels = (test_scores[high_confidence_idx] >= 0).astype(int)
        pseudo_labels = test_labels[high_confidence_idx]

        pseudo_features = test_features[high_confidence_idx]
        
        num_positive = np.sum(pseudo_labels)  
        num_negative = len(pseudo_labels) - num_positive
        total_selected = len(pseudo_labels)

        print(f"Total selected samples: {total_selected}")
        print(f"Positive class (1): {num_positive} ({num_positive / total_selected:.2%})")
        print(f"Negative class (0): {num_negative} ({num_negative / total_selected:.2%})")
        
        test_f1, train_f1, acc, train_acc, test_loss, train_loss = error.evaluation_metrics(f"Naive Iteration {i}", svcModel, pseudo_features, train_features, pseudo_labels, train_labels)

    
if __name__=="__main__":
    main()
