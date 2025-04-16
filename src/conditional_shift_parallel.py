from joblib import Parallel, delayed

from argparse import ArgumentParser
import os
import CommonModules as CM
from sklearn.feature_extraction.text import TfidfVectorizer as TF
import numpy as np
from scipy.sparse import vstack
from sklearn.utils import shuffle
import gc
from sklearn.svm import SVC
from scipy.spatial.distance import jensenshannon
import pickle


def compute_js_by_year(years, models, x, year):
    js_vectors = {}
    js_vectors[year] = {}
    probs_by_year = {}
    for year in years:
        clf = models[year]
        probs = clf.predict_proba(x)
        probs_by_year[year] = probs

    # JS divergence matrix: divergence[i,j] = JS(P_i || P_j on data from year_i)
    for year_j in years:
        if(year_j==year):
            continue  
        js_divs = []
        probs_i = probs_by_year[year]
        probs_j = probs_by_year[year_j]
        for pi, pj in zip(probs_i, probs_j):
            js = jensenshannon(pi, pj, base=2)**2
            js_divs.append(js)
        js_vectors[year][year_j] = np.mean(js_divs)
        del js_divs
        gc.collect()

    del probs_by_year
    gc.collect()

    return js_vectors 

def getFeature(dir, years):
    MalwareCorpus = [os.path.join(dir, "training", "malware"), os.path.join(dir, "test", "malware")]
    GoodwareCorpus = [os.path.join(dir, "training", "goodware"), os.path.join(dir, "test", "goodware")]
    AllMalSamples = CM.ListFiles(MalwareCorpus, ".data", year=years)
    AllGoodSamples = CM.ListFiles(GoodwareCorpus, ".data", year=years)
    return AllMalSamples, AllGoodSamples


def train_svc_models(year, dir, featureVectorizer):
    malwares, goodwares = getFeature(dir, [year])
    goodfeatures = featureVectorizer.transform(goodwares)
    malfeatures = featureVectorizer.transform(malwares)
    train_features = vstack([goodfeatures, malfeatures])
    train_labels = np.hstack([np.zeros(goodfeatures.shape[0]), np.ones(malfeatures.shape[0])])
    train_features, train_labels = shuffle(train_features, train_labels, random_state=2314)
    svcModel = SVC(kernel='rbf', C=1, gamma='scale', probability=True)
    svcModel.fit(train_features, train_labels)
    del goodfeatures, malfeatures, train_labels
    gc.collect()
    return year, svcModel, train_features



if __name__=="__main__":
    dir = "/home/wang/Data/android"

    years = [str(i) for i in range(2014, 2015)]
    allmals, allgoods = getFeature(dir, years)
    featureVectorizer = TF(input="filename", tokenizer=lambda x: x.split('\n'), token_pattern=None, binary=True)
    featureVectorizer.fit(allmals+allgoods)
    del allmals, allgoods
    gc.collect()

    x_features = {}
    models = {}
    results = Parallel(n_jobs=2)(
        delayed(train_svc_models)(year, dir, featureVectorizer) for year in years
    )
    models = {year: model for year, model, _ in results}
    x_features = {year: feats for year, _, feats in results}
    
    divergencies = Parallel(n_jobs=2)(
        delayed(compute_js_by_year)(years, models, x_features[year], year) for year in years
    )
    print(divergencies)
    with open("/home/wang/Data/divergencies.pkl", "wb") as f:
        pickle.dump(divergencies, f)