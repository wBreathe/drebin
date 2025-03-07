import os
import CommonModules as CM
from sklearn.feature_extraction.text import TfidfVectorizer as TF
import numpy as np
from scipy.sparse import vstack

def check_overlap(features1, features2, name1, name2):
    features1_arr = features1.toarray()
    features2_arr = features2.toarray()
    matches = (features1_arr[:, None] == features2_arr).all(-1)
    overlap_count = np.count_nonzero(matches)
    print(f"Overlap between {name1} and {name2}: {overlap_count}")
    return matches

def get_unique_samples(set1, set2, name):
    set1 = set1.toarray()
    set2 = set2.toarray()
    index =  ~np.any((set1[:, None] == set2).all(-1), axis=1)
    unique_samples = set1[index]
    print(f"uniques samples in {name}: {len(unique_samples)}")
    return unique_samples

def getFeature(dir, years):
    MalwareCorpus=os.path.join(dir, "training", "malware"),
    GoodwareCorpus=os.path.join(dir, "training", "goodware"),
    AllMalSamples = CM.ListFiles(MalwareCorpus, ".data", year=years)
    AllGoodSamples = CM.ListFiles(GoodwareCorpus, ".data", year=years)
    return AllMalSamples, AllGoodSamples

def main():
    dir = "/home/wang/Data/android"
    print("***********************imbalance************************")
    train_years = [str(i) for i in range(2010,2016)]
    test_years = [str(i) for i in range(2016, 2024)]
    malwares, goodwares = getFeature(dir, train_years)
    tmalwares, tgoodwares = getFeature(dir, test_years)
    print("imbalance, train: ", len(malwares)/len(goodwares), "; test: ", len(tmalwares)/len(tgoodwares))
    FeatureVectorizer = TF(input='filename', tokenizer=lambda x: x.split('\n'), token_pattern=None,binary=True)
    FeatureVectorizer.fit(malwares+goodwares)
    goodfeatures = FeatureVectorizer.transform(goodwares)
    malfeatures = FeatureVectorizer.transform(malwares)
    tgoodfeatures = FeatureVectorizer.transform(tgoodwares)
    tmalfeatures = FeatureVectorizer.transform(tmalwares)
    print("***********************Ps(Y|X) & Pt(Y|X)************************")
    print(tgoodfeatures.shape, tmalfeatures.shape)
    overlaps_a = check_overlap(set(goodfeatures), set(tmalfeatures), "goodwares", "tmalwares")
    overlaps_b = check_overlap(set(malfeatures), set(tgoodfeatures), "malwares", "tgoodwares")
    print('add everything')
    NewFeatureVectorizer = TF(input='filename', tokenizer=lambda x: x.split('\n'), token_pattern=None,binary=True)
    NewFeatureVectorizer.fit(malwares+goodwares+tmalwares+tgoodwares)
    newgoodfeatures = NewFeatureVectorizer.transform(goodwares)
    newmalfeatures = NewFeatureVectorizer.transform(malwares)
    newtgoodfeatures = NewFeatureVectorizer.transform(tgoodwares)
    newtmalfeatures = NewFeatureVectorizer.transform(tmalwares)
    print(newtgoodfeatures.shape, newtmalfeatures.shape)
    overlaps_na = check_overlap(set(newgoodfeatures), set(newtmalfeatures), "goodwares", "tmalwares")
    overlaps_nb = check_overlap(set(newmalfeatures), set(newtgoodfeatures), "malwares", "tgoodwares")
    print("***********************supp(S) & supp(T)************************")
    train_features = vstack([goodfeatures, malfeatures])
    test_features = vstack([tgoodfeatures, tmalfeatures])
    new_train_features = vstack([newgoodfeatures, newmalfeatures])
    new_test_features = vstack([newtgoodfeatures, newtmalfeatures])
    not_in_test = get_unique_samples(train_features,test_features, "train (old)")
    not_in_train = get_unique_samples(test_features, train_features, "test (old)")
    new_not_in_test = get_unique_samples(new_train_features, new_test_features, "train (new)")
    new_not_in_train = get_unique_samples(new_test_features, new_train_features, "test (new)")


if __name__=="__main__":
    main()
