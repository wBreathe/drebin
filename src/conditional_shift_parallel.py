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
from sklearn.model_selection import train_test_split

def compute_js_by_year(years, models, x, label, year):
    x = x.copy()
    x.sort_indices()
    js_vectors = {}
    js_vectors[year] = {}
    probs_by_year = {}
    acc_by_year = {}
    acc_by_year[year] = {}
    for y in years:
        clf = models[y]
        probs = clf.predict_proba(x)
        preds = clf.predict(x)
        acc = np.mean(preds==label)
        acc_by_year[year][y] = acc
        probs_by_year[y] = probs

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

    return js_vectors, acc_by_year

def getFeature(dir, years):
    MalwareCorpus = [os.path.join(dir, "training", "malware"), os.path.join(dir, "test", "malware")]
    GoodwareCorpus = [os.path.join(dir, "training", "goodware"), os.path.join(dir, "test", "goodware")]
    AllMalSamples = CM.ListFiles(MalwareCorpus, ".data", year=years)
    AllGoodSamples = CM.ListFiles(GoodwareCorpus, ".data", year=years)
    return AllMalSamples, AllGoodSamples


def train_svc_models(year, dir, featureVectorizer):
    malwares, goodwares = getFeature(dir, [year])
    print(year, len(malwares), len(goodwares))
    goodfeatures = featureVectorizer.transform(goodwares)
    malfeatures = featureVectorizer.transform(malwares)
    all_features = vstack([goodfeatures, malfeatures])
    all_labels = np.hstack([np.zeros(goodfeatures.shape[0]), np.ones(malfeatures.shape[0])])
    X_train, X_test, y_train, y_test = train_test_split(
        all_features, all_labels,
        test_size=0.2,
        stratify=all_labels,
        random_state=2314
    )

    svcModel = SVC(kernel='rbf', C=1, gamma='scale', probability=True)
    svcModel.fit(X_train, y_train)
    del goodfeatures, malfeatures, all_features, all_labels, X_train, y_train
    gc.collect()
    return year, svcModel, X_test, y_test


if __name__=="__main__":
    dir = "/home/wang/Data/android"

    years = [str(i) for i in range(2012, 2023)]
    allmals, allgoods = getFeature(dir, years)
    featureVectorizer = TF(input="filename", tokenizer=lambda x: x.split('\n'), token_pattern=None, binary=True)
    featureVectorizer.fit(allmals+allgoods)
    del allmals, allgoods
    gc.collect()

    x_features = {}
    models = {}
    y_labels = {}
    results = Parallel(n_jobs=10)(
        delayed(train_svc_models)(year, dir, featureVectorizer) for year in years
    )
    models = {year: model for year, model, _, _ in results}
    x_features = {year: feats for year, _, feats, _ in results}
    y_labels = {year: labels for year, _, _, labels in results}
    results = Parallel(n_jobs=10)(
        delayed(compute_js_by_year)(years, models, x_features[year], y_labels[year], year) for year in years
    )
    
    
    all_js = {}
    all_acc = {}

    for js_vec, acc_vec in results:
        for from_year in js_vec:
            if from_year not in all_js:
                all_js[from_year] = {}
            for to_year, js_val in js_vec[from_year].items():
                all_js[from_year][to_year] = js_val

        for from_year in acc_vec:
            if from_year not in all_acc:
                all_acc[from_year] = {}
            for to_year, acc_val in acc_vec[from_year].items():
                all_acc[from_year][to_year] = acc_val
    
    with open("/home/wang/Data/drift_results.pkl", "wb") as f:
        pickle.dump({"js": all_js, "acc": all_acc}, f)
    
    print(all_js)
    print(all_acc)