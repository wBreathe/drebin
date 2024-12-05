import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer as TF
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score
import logging
import random
import CommonModules as CM
import joblib
#from pprint import pprint
import json, os
import pickle
import random
from numpy.linalg import norm
import error

logging.basicConfig(level=logging.INFO)
Logger = logging.getLogger('RandomClf.stdout')
Logger.setLevel("INFO")


def RandomClassification(dual, penalty, years, enable_imbalance, MalwareCorpus, GoodwareCorpus, TestSize, FeatureOption, Model, NumTopFeats, saveTrainSet=""):
    '''
    Train a classifier for classifying malwares and goodwares using Support Vector Machine technique.
    Compute the prediction accuracy and f1 score of the classifier.
    Modified from Jiachun's code.

    :param String MalwareCorpus: absolute path of the malware corpus
    :param String GoodwareCorpus: absolute path of the goodware corpus
    :param String FeatureOption: tfidf or binary, specify how to construct the feature vector

    :rtype String Report: result report
    '''
    # step 1: creating feature vector
    print("PART RANDOM")
    Logger.debug("Loading Malware and Goodware Sample Data")
    label = f"_dual-{dual}_penalty-{penalty}"
    AllMalSamples = CM.ListFiles(MalwareCorpus, ".data", year=years)
    AllGoodSamples = CM.ListFiles(GoodwareCorpus, ".data", year=years)
    if(enable_imbalance):
        AllMalSamples = random.sample(AllMalSamples, int(0.2*len(AllGoodSamples))) if len(AllMalSamples)>int(0.2*len(AllGoodSamples)) else AllMalSamples
    AllSampleNames = AllMalSamples + AllGoodSamples
    Logger.info("Loaded samples")

    FeatureVectorizer = TF(input='filename', tokenizer=lambda x: x.split('\n'), token_pattern=None,
                           binary=FeatureOption)
    x = FeatureVectorizer.fit_transform(AllMalSamples + AllGoodSamples)
    with open(os.path.join(saveTrainSet, "featureVector_{label}.pkl"), "wb") as f:
        pickle.dump(AllMalSamples+AllGoodSamples, f)
    print(f"dimension of features: {x.shape}")

    # label malware as 1 and goodware as -1
    Mal_labels = np.ones(len(AllMalSamples))
    Good_labels = np.empty(len(AllGoodSamples))
    Good_labels.fill(-1)
    y = np.concatenate((Mal_labels, Good_labels), axis=0)
    Logger.info("Label array - generated")


    # step 2: split all samples to training set and test set
    x_train_samplenames, x_test_samplenames, y_train, y_test = train_test_split(AllSampleNames, y, test_size=TestSize,
                                                     random_state=random.randint(0, 100), stratify=y)
    
    x_train = FeatureVectorizer.fit_transform(x_train_samplenames)
    x_test = FeatureVectorizer.transform(x_test_samplenames)
    Logger.debug("Test set split = %s", TestSize)
    Logger.info("train-test split done")
    if(saveTrainSet!=""):
        with open(os.path.join(saveTrainSet,f"trainSamples_{label}.pkl"),"wb") as f:
            pickle.dump((x_train_samplenames, y_train), f)
    
    # step 3: train the model
    Logger.info("Perform Classification with SVM Model")
    Parameters= {'C': [0.01, 0.1, 1, 10, 100]}
    print(f"number of samples in training set: {x_train.shape}, number of samples in test set: {x_test.shape}")
    T0 = time.time() 
    if not Model:
        Clf = GridSearchCV(LinearSVC(max_iter=1000000,dual=dual, penalty=penalty, fit_intercept=False), Parameters, cv= 5, scoring= 'f1', n_jobs=-1 )
        SVMModels= Clf.fit(x_train, y_train)
        Logger.info("Processing time to train and find best model with GridSearchCV is %s sec." %(round(time.time() -T0, 2)))
        BestModel= SVMModels.best_estimator_
        Logger.info("Best Model Selected : {}".format(BestModel))
        print(("The training time for random split classification is %s sec." % (round(time.time() - T0,2))))
        filename = "randomClassification"
        joblib.dump(Clf, filename + f"_{label}.pkl")
    else:
        SVMModels = joblib.load(Model)
        BestModel= SVMModels.best_estimator

    # step 4: Evaluate the best model on test set
    print("Start evaluation ......")
    T0 = time.time()
    y_pred = SVMModels.predict(x_test)
    y_train_pred = SVMModels.predict(x_train)
    print(("The testing time for random split classification is %s sec." % (round(time.time() - T0,2))))
    Accuracy = f1_score(y_test, y_pred, average='binary')
    print(("Test Set F1 = {}".format(Accuracy)))
    Train_Accuracy = f1_score(y_train, y_train_pred, average='binary')
    print(("Train Set F1 = {}".format(Train_Accuracy)))
    print((metrics.classification_report(y_test,
                                         y_pred, labels=[1, -1],
                                         target_names=['Malware', 'Goodware'])))
    Report = "Test Set F1 = " + str(Accuracy) + "\n" + metrics.classification_report(y_test,
                                                                                           y_pred,
                                                                                           labels=[1, -1],
                                                                                           target_names=['Malware',
                                                                                                         'Goodware'])
    # pointwise multiplication between weight and feature vect
    print(f"iteration in sum: {BestModel.n_iter_}")
    all_parameters = np.prod(BestModel.coef_.shape)
    print(f"all parameters: {all_parameters}")
    
    w = BestModel.coef_
    l1_norm = norm(w, ord=1)
    l2_norm = norm(w)
    print(f"C:{BestModel.C}")
    print(f"weights: {w}")
    print(f"l1 norm:{l1_norm}, l2 norm:{l2_norm}")

    print("Calculating loss ....")
    num_samples = 50
    sampled_w = error.sample_spherical_gaussian_from_w(w, 0.1, num_samples)
    avg_loss, std_loss = error.get_loss(BestModel, sampled_w, x_train, y_train)
    print(f"loss results for training set for {num_samples} samples: {avg_loss}±{std_loss}. ")
    sampled_w = error.sample_spherical_gaussian_from_w(w, 0.1, num_samples)
    avg_loss, std_loss = error.get_loss(BestModel, sampled_w, x_test, y_test)
    print(f"loss results for test set for {num_samples} samples: {avg_loss}±{std_loss}. ")

    '''
    print("Start interpreting ....")
    w = w[0].tolist()
    v = x_test.toarray()
    vocab = FeatureVectorizer.get_feature_names_out()
    explanations = {os.path.basename(s):{} for s in x_test_samplenames}
    for i in range(v.shape[0]):
        wx = v[i, :] * w
        wv_vocab = list(zip(wx, vocab))
        if y_pred[i] == 1:
            wv_vocab.sort(reverse=True)
            explanations[os.path.basename(x_test_samplenames[i])]['top_features'] = wv_vocab[:NumTopFeats]
        elif y_pred[i] == -1:
            wv_vocab.sort()
            explanations[os.path.basename(x_test_samplenames[i])]['top_features'] = wv_vocab[-NumTopFeats:]
        explanations[os.path.basename(x_test_samplenames[i])]['original_label'] = y_test[i]
        explanations[os.path.basename(x_test_samplenames[i])]['predicted_label'] = y_pred[i]
   
    with open(f'explanations_RC_{label}.json','w') as FH:
        json.dump(explanations,FH,indent=4)
    '''

    return Report
