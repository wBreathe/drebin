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
import argparse

def getFeature(dir, years):
    MalwareCorpus=os.path.join(dir, "training", "malware")
    GoodwareCorpus=os.path.join(dir, "training", "goodware")
    AllMalSamples = CM.ListFiles(MalwareCorpus, ".data", year=years)
    AllGoodSamples = CM.ListFiles(GoodwareCorpus, ".data", year=years)
    return AllMalSamples, AllGoodSamples


def main():
    dir = "/home/wang/Data/android"
    Args =  argparse.ArgumentParser(description="Check the distribution")
    Args.add_argument("--N", dest="n", type=int, default=100, required=True)
    args = Args.parse_args()
    device = torch.device("cpu")
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
    
    train_labels = np.hstack([np.zeros(goodfeatures.shape[0]), np.ones(malfeatures.shape[0])])
    test_labels = np.hstack([np.zeros(tgoodfeatures.shape[0]), np.ones(tmalfeatures.shape[0])])
    train_features, train_labels = shuffle(train_features, train_labels, random_state=2314)
    test_features, test_labels = shuffle(test_features, test_labels, random_state=2314)
    svcModel = SVC(kernel='rbf', C=1, gamma='scale')
    svcModel.fit(train_features, train_labels)
    test_f1, train_f1, acc, train_acc, test_loss, train_loss = error.evaluation_metrics(f"Support shift initial", svcModel, test_features, train_features, test_labels, train_labels)

    # start sampling n closest test samples
    decision_values = np.abs(svcModel.decision_function(test_features))
    closest_indices = np.argsort(decision_values)[: args.n]
    new_train_features = test_features[closest_indices]
    new_train_labels = test_labels[closest_indices]
    mask = np.ones(test_features.shape[0], dtype=bool)
    mask[closest_indices] = False
    remaining_test_features = test_features[mask]
    remaining_test_labels = test_labels[mask]

    # Update training set
    train_features = vstack([train_features, new_train_features])
    train_labels = np.hstack([train_labels, new_train_labels])

    # Shuffle updated training set
    train_features, train_labels = shuffle(train_features, train_labels, random_state=2314)

    # Retrain SVM model
    svcModel = SVC(kernel='rbf', C=1, gamma='scale')
    svcModel.fit(train_features, train_labels)

    # Evaluate new model on remaining test set
    error.evaluation_metrics(
        f"After adding N closest points on new test and new train with N={args.n}", svcModel, remaining_test_features, train_features, remaining_test_labels, train_labels
    )


if __name__=="__main__":
    main()