import numpy as np
import time
import CommonModules as CM
from sklearn.feature_extraction.text import TfidfVectorizer as TF
from sklearn.svm import LinearSVC,SVC
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score
import logging
import joblib
import json, os
#from pprint import pprint
import pickle
import random
from numpy.linalg import norm
import error
from error import HoldoutConfig

logging.basicConfig(level=logging.INFO)
Logger = logging.getLogger('HoldoutClf.stdout')
Logger.setLevel("INFO")


def HoldoutClassification(i:int, model, rounded:int, config: HoldoutConfig):
    '''
    Train a classifier for classifying malwares and goodwares using Support Vector Machine technique.
    Compute the prediction accuracy and f1 score of the classifier.
    Modified from Jiachun's code.

    :param String/List TrainMalSet: absolute path/paths of the malware corpus for trainning set
    :param String/List TrainGoodSet: absolute path/paths of the goodware corpus for trainning set
    :param String/List TestMalSet: absolute path/paths of the malware corpus for test set
    :param String/List TestGoodSet: absolute path/paths of the goodware corpus for test set
    :param String FeatureOption: tfidf or binary, specify how to construct the feature vector
    '''
    print("PART HOLDOUT")
    kernel = config.kernel
    NCpuCores = config.NCpuCores
    priorPortion = config.priorPortion
    dual = config.dual
    penalty = config.penalty
    years = config.years
    saveTrainSet = config.saveTrainSet
    enable_imbalance = config.enable_imbalance
    TestMalSet = config.TestMalSet
    TestGoodSet = config.TestGoodSet
    TestSize = config.TestSize
    FeatureOption = config.FeatureOption
    Model = config.Model
    NumTopFeats = config.NumTopFeats

    # step 1: creating feature vector
    label = f"_partition_num-{i}_kernel-{kernel}_testSize-{TestSize}_priorPortion-{priorPortion}"
    Logger.debug("Loading Malware and Goodware Sample Data for training and testing")
    with open(os.path.join(saveTrainSet,f"trainSamples_{label}.pkl"), 'rb') as f:
        x_train_names, y_train = pickle.load(f)
    # if(priorPortion!=0):
    #     with open(os.path.join(saveTrainSet,f"priorSamples_{label}.pkl"), 'rb') as f:
    #         x_train_prior_names, y_train_prior = pickle.load(f)
    # sample_num = int(len(y_train)/(1-TestSize)*TestSize)
    # malList = CM.ListFiles(TestMalSet, ".data", year=years)
    # goodList = CM.ListFiles(TestGoodSet, ".data", year=years)
    # TestMalSamples = random.sample(malList, sample_num) if len(malList)>sample_num else malList
    # TestGoodSamples = random.sample(goodList, sample_num) if len(goodList)>sample_num else goodList
    TestMalSamples = CM.ListFiles(TestMalSet, ".data", year=years)
    TestGoodSamples = CM.ListFiles(TestGoodSet, ".data", year=years)
    if(enable_imbalance):
        TestMalSamples = random.sample(TestMalSamples, int(0.2*len(TestGoodSamples))) if len(TestMalSamples)>int(0.2*len(TestGoodSamples)) else TestMalSamples
    AllTestSamples = TestMalSamples + TestGoodSamples
    print("all test samples, test good samples, test malware samples", len(AllTestSamples), len(TestGoodSamples), len(TestMalSamples))
    FeatureVectorizer = TF(input="filename", tokenizer=lambda x: x.split('\n'), token_pattern=None,
                           binary=FeatureOption)
    
    with open(os.path.join(saveTrainSet, f"featureVector_{label}.pkl"), "rb") as f:
        allSamples = pickle.load(f)
    FeatureVectorizer.fit(allSamples)
    x_train = FeatureVectorizer.transform(x_train_names)
    x_test = FeatureVectorizer.transform(TestMalSamples + TestGoodSamples)

    Logger.info("Training Label array - generated")

    Test_Mal_labels = np.ones(len(TestMalSamples))
    Test_Good_labels = np.empty(len(TestGoodSamples))
    Test_Good_labels.fill(-1)
    y_test = np.concatenate((Test_Mal_labels, Test_Good_labels), axis=0)
    Logger.info("Testing Label array - generated")

    # step 2: train the model
    Logger.info("Perform Classification with SVM Model")
    print(f"number of samples in training set: {x_train.shape[0]}, number of samples in test set: {x_test.shape[0]}")
    T0 = time.time()
    # if not Model:
    #     if(priorPortion!=0):
    #         x_train_prior = FeatureVectorizer.transform(x_train_prior_names)
    #         PriorModel = LinearSVC(max_iter=1000000, dual=dual, penalty=penalty, C=1, fit_intercept=False)
    #         PriorModel.fit(x_train_prior, y_train_prior)
    #     BestModel = LinearSVC(max_iter=1000000, dual=dual, penalty=penalty, C=1, fit_intercept=False)
    #     BestModel.fit(x_train, y_train)
        # filename = "houldoutClassification"
        # joblib.dump(Clf, filename+f"_{label}_holdout.pkl")
    # else:
        # # SVMModels= joblib.load(Model)
        # BestModel = joblib.load(Model)
        # # BestModel= SVMModels.best_estimator_
        # TrainingTime = 0
    # Logger.info("shape", x_train.shape, x_test)
    # step 4: Evaluate the best model on test set
    # w = BestModel.coef_
    # w_norm = w/norm(w)
    step = rounded * 0.01
    mu_values = [rounded + step * i for i in range(-5, 6)]
    results = []
    for mu in mu_values:
        pacc,ptrain_acc,ptest_f1, ptrain_f1, ptest_loss, ptrain_loss = 0,0,0,0,0,0
        # test_f1, train_f1, acc, train_acc, test_loss, train_loss = error.evaluation_metrics(f"random classification with priorportion-{priorPortion}", BestModel, x_test, x_train, y_test, y_train)
        BestModel = model
        BestModel.coef_ = mu*model.coef_
        test_f1, train_f1, acc, train_acc, test_loss, train_loss = error.evaluation_metrics(f"random classification with normed priorportion-{priorPortion}", BestModel, x_test, x_train, y_test, y_train)
        results.append([i, mu, test_f1, train_f1, acc, train_acc, test_loss, train_loss])
    # print(f"Calculating loss with priorportion-{priorPortion}....")
    # num_samples = 100
    # sampled_w = error.sample_spherical_gaussian_from_w(w, num_samples)
    # print(f"get {num_samples} weight distribution using for training set")
    # avg_loss, std_loss = error.get_loss_multiprocessing(BestModel, sampled_w, x_train, y_train,NCpuCores)
    # print(f"loss results for training set for {num_samples} weights: {avg_loss}±{std_loss}. ")
    # sampled_w = error.sample_spherical_gaussian_from_w(w, num_samples)
    # print(f"get {num_samples} weight distribution using for test set")
    # avg_loss, std_loss = error.get_loss_multiprocessing(BestModel, sampled_w, x_test, y_test,NCpuCores)
    # print(f"loss results for test set for {num_samples} weights: {avg_loss}±{std_loss}. ")
    

    '''
    print("Start interpreting ....")
    w = w[0].tolist()
    v = x_test.toarray()
    vocab = FeatureVectorizer.get_feature_names_out()
    explanations = {os.path.basename(s):{} for s in AllTestSamples}
    for i in range(v.shape[0]):
        wx = v[i, :] * w
        wv_vocab = list(zip(wx, vocab))
        if y_pred[i] == 1:
            wv_vocab.sort(reverse=True)
            explanations[os.path.basename(AllTestSamples[i])]['top_features'] = wv_vocab[:NumTopFeats]
        elif y_pred[i] == -1:
            wv_vocab.sort()
            explanations[os.path.basename(AllTestSamples[i])]['top_features'] = wv_vocab[-NumTopFeats:]
        explanations[os.path.basename(AllTestSamples[i])]['original_label'] = y_test[i]
        explanations[os.path.basename(AllTestSamples[i])]['predicted_label'] = y_pred[i]

    with open(f'explanations_HC_{label}.json','w') as FH:
        json.dump(explanations,FH,indent=4)
    '''
    return results
    # return {'f1_test':test_f1, 'f1_train':train_f1, 'acc_test':acc, "acc_train":train_acc, 'loss_test':test_loss, "loss_train":train_loss, "f1_test_prior":ptest_f1, "f1_train_prior": ptrain_f1, "acc_test_prior":pacc, "acc_train_prior":ptrain_acc,"loss_test_prior":ptest_loss, "loss_train_prior":ptrain_loss, "l1_norm":l1_norm, "l2_norm":l2_norm, "full":full}