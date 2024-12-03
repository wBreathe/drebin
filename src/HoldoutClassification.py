import numpy as np
import time
import CommonModules as CM
from sklearn.feature_extraction.text import TfidfVectorizer as TF
from sklearn.svm import LinearSVC
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

logging.basicConfig(level=logging.INFO)
Logger = logging.getLogger('HoldoutClf.stdout')
Logger.setLevel("INFO")


def HoldoutClassification(label, years, saveTrainSet, enable_imbalance, TestMalSet, TestGoodSet, TestSize, FeatureOption, Model, NumTopFeats):
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
    # step 1: creating feature vector
    Logger.debug("Loading Malware and Goodware Sample Data for training and testing")
    with open(os.path.join(saveTrainSet,f"trainSamples_{label}.pkl"), 'rb') as f:
        x_train_names, y_train = pickle.load(f)
    # print("hello:", len(y_train), len(CM.ListFiles(TestMalSet, ".data", year=years)), len(CM.ListFiles(TestGoodSet, ".data", year=years)), int(len(y_train)/(1-TestSize)*TestSize))
    sample_num = int(len(y_train)/(1-TestSize)*TestSize)
    malList = CM.ListFiles(TestMalSet, ".data", year=years)
    goodList = CM.ListFiles(TestGoodSet, ".data", year=years)
    TestMalSamples = random.sample(malList, sample_num) if len(malList)>sample_num else malList
    TestGoodSamples = random.sample(goodList, sample_num) if len(goodList)>sample_num else goodList
    if(enable_imbalance):
        TestMalSamples = random.sample(TestMalSamples, int(0.2*len(TestGoodSamples))) if len(TestMalSamples)>int(0.2*len(TestGoodSamples)) else TestMalSamples
    AllTestSamples = TestMalSamples + TestGoodSamples
    print(len(AllTestSamples), len(TestGoodSamples), len(TestMalSamples))
    Logger.info("Loaded Samples")

    FeatureVectorizer = TF(input="filename", tokenizer=lambda x: x.split('\n'), token_pattern=None,
                           binary=FeatureOption)
    # with open(os.path.join(saveTrainSet,"featureVector.pkl"), "rb") as f:
    #     all_samples = pickle.load(f)
    x_train = FeatureVectorizer.fit_transform(x_train_names)
    # x_train = FeatureVectorizer.fit_transform(TrainMalSamples + TrainGoodSamples)
    x_test = FeatureVectorizer.transform(TestMalSamples + TestGoodSamples)

    # label training sets malware as 1 and goodware as -1
    # Train_Mal_labels = np.ones(len(TrainMalSamples))
    # Train_Good_labels = np.empty(len(TrainGoodSamples))
    # Train_Good_labels.fill(-1)
    # y_train = np.concatenate((Train_Mal_labels, Train_Good_labels), axis=0)
    Logger.info("Training Label array - generated")

    # label testing sets malware as 1 and goodware as -1
    Test_Mal_labels = np.ones(len(TestMalSamples))
    Test_Good_labels = np.empty(len(TestGoodSamples))
    Test_Good_labels.fill(-1)
    y_test = np.concatenate((Test_Mal_labels, Test_Good_labels), axis=0)
    Logger.info("Testing Label array - generated")

    # step 2: train the model
    Logger.info("Perform Classification with SVM Model")
    Parameters= {'C': [0.01, 0.1, 1, 10, 100]}
    print(f"number of samples in training set: {x_train.shape[0]}, number of samples in test set: {x_test.shape[0]}")
    T0 = time.time()
    if not Model:
        Clf = GridSearchCV(LinearSVC(max_iter=1000000, dual=False, penalty='l2'), Parameters, cv= 5, scoring= 'f1', n_jobs=-1 )
        SVMModels= Clf.fit(x_train, y_train)
        Logger.info("Processing time to train and find best model with GridSearchCV is %s sec." %(round(time.time() -T0, 2)))
        BestModel= SVMModels.best_estimator_
        Logger.info("Best Model Selected : {}".format(BestModel))
        TrainingTime = round(time.time() - T0,2)
        print(("The training time for random split classification is %s sec." % (TrainingTime)))
        # print("Enter a filename to save the model:")
        filename = "houldoutClassification"
        joblib.dump(Clf, filename+f"_{label}_holdout.pkl")
    else:
        SVMModels= joblib.load(Model)
        BestModel= SVMModels.best_estimator_
        TrainingTime = 0
    print("shape", x_train.shape, x_test)
    # step 4: Evaluate the best model on test set
    y_pred = SVMModels.predict(x_test)
    y_train_pred = SVMModels.predict(x_train)
    TestingTime = round(time.time() - TrainingTime - T0,2)
    Accuracy = f1_score(y_test, y_pred, average='binary')  # Return (x1 == x2) element-wise.
    TrainAccuracy = f1_score(y_train,y_train_pred, average='binary')
    print(("Test Set F1 = ", Accuracy))
    print(("Train Set F1 = ", TrainAccuracy))
    print((metrics.classification_report(y_test,
                                        y_pred, labels=[1, -1],
                                        target_names=['Malware', 'Goodware'])))
    Report = "Test Set F1 = " + str(Accuracy) + "\n" + metrics.classification_report(y_test,
                                                                                           y_pred,
                                                                                           labels=[1, -1],
                                                                                           target_names=['Malware',
                                                                                                         'Goodware'])
    # pointwise multiplication between weight and feature vect
    print(f"actual number of iterations: {BestModel.n_iter_}")
    num_parameters = np.prod(BestModel.coef_.shape) + BestModel.intercept_.size
    print(f"number of parameters: {num_parameters}")
    w = BestModel.coef_
    l1_norm = norm(w, ord=1)
    l2_norm = norm(w)
    print(f"weights:{w}")
    print(f"l1_norm:{l1_norm}, l2_norm:{l2_norm}")
    w = w[0].tolist()
    v = x_test.toarray()
    vocab = FeatureVectorizer.get_feature_names_out()
    explanations = {os.path.basename(s):{} for s in AllTestSamples}
    for i in range(v.shape[0]):
        wx = v[i, :] * w
        wv_vocab = list(zip(wx, vocab))
        if y_pred[i] == 1:
            wv_vocab.sort(reverse=True)
           # print("pred: {}, org: {}".format(y_pred[i],y_test[i]))
           # pprint(wv_vocab[:10])
            explanations[os.path.basename(AllTestSamples[i])]['top_features'] = wv_vocab[:NumTopFeats]
        elif y_pred[i] == -1:
            wv_vocab.sort()
           # print("pred: {}, org: {}".format(y_pred[i],y_test[i]))
           # pprint(wv_vocab[-10:])
            explanations[os.path.basename(AllTestSamples[i])]['top_features'] = wv_vocab[-NumTopFeats:]
        explanations[os.path.basename(AllTestSamples[i])]['original_label'] = y_test[i]
        explanations[os.path.basename(AllTestSamples[i])]['predicted_label'] = y_pred[i]

    with open(f'explanations_HC_{label}.json','w') as FH:
        json.dump(explanations,FH,indent=4)

    return y_train, y_test, y_pred, TrainingTime, TestingTime
