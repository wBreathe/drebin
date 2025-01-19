import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer as TF
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC, SVC
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
from error import RandomConfig


logging.basicConfig(level=logging.INFO)
Logger = logging.getLogger('RandomClf.stdout')
Logger.setLevel("INFO")


def RandomClassification(num:int, config: RandomConfig):
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
    kernel = config.kernel
    NCpuCores = config.NCpuCores
    priorPortion = config.priorPortion
    eta = config.eta
    dual = config.dual
    penalty = config.penalty
    years = config.years
    enable_imbalance = config.enable_imbalance
    MalwareCorpus = config.MalwareCorpus
    GoodwareCorpus = config.GoodwareCorpus
    TestSize = config.TestSize
    FeatureOption = config.FeatureOption
    Model = config.Model
    NumTopFeats = config.NumTopFeats
    saveTrainSet = config.saveTrainSet
    enableFuture = config.enableFuture
    futureYears = config.futureYears
    futureMalwareCorpus = config.futureMalwareCorpus
    futureGoodwareCorpus = config.futureGoodwareCorpus
    
    print("PART RANDOM")
    Logger.debug("Loading Malware and Goodware Sample Data")
    label = f"_eta-{eta}_num-{num}_kernel-{kernel}_testSize-{TestSize}_priorPortion-{priorPortion}"
    AllMalSamples = CM.ListFiles(MalwareCorpus, ".data", year=years)
    AllGoodSamples = CM.ListFiles(GoodwareCorpus, ".data", year=years)
    print("number of samples: ", len(AllMalSamples), len(AllGoodSamples))

    FeatureVectorizer = TF(input='filename', tokenizer=lambda x: x.split('\n'), token_pattern=None,binary=FeatureOption)
    if(enableFuture):
        assert((futureYears is not None) and (futureMalwareCorpus is not None) and (futureGoodwareCorpus is not None))
        FutureMalSamples = CM.ListFiles(futureMalwareCorpus, ".data", year=futureYears)
        FutureGoodSamples = CM.ListFiles(futureGoodwareCorpus, ".data", year=futureYears)
        FeatureVectorizer.fit(AllMalSamples + AllGoodSamples + FutureGoodSamples + FutureMalSamples)
        with open(os.path.join(saveTrainSet, f"featureVector_{label}.pkl"), "wb") as f:
            pickle.dump(AllMalSamples+AllGoodSamples+FutureGoodSamples+FutureMalSamples, f)
    
    if(enable_imbalance):
        # AllMalSamples = random.sample(AllMalSamples, int(0.2*len(AllGoodSamples))) if len(AllMalSamples)>int(0.2*len(AllGoodSamples)) else AllMalSamples
        num_good_samples = (len(AllMalSamples) / TestSize)  
        if(num_good_samples > len(AllGoodSamples)):
            raise Exception("Error: Not enough good wares!")
        AllGoodSamples = random.sample(AllGoodSamples, num_good_samples)
    AllSampleNames = AllMalSamples + AllGoodSamples
    Logger.info("Loaded samples")
    
    if(not enableFuture):
        FeatureVectorizer.fit(AllMalSamples + AllGoodSamples)
        with open(os.path.join(saveTrainSet, f"featureVector_{label}.pkl"), "wb") as f:
            pickle.dump(AllMalSamples+AllGoodSamples, f)

    # label malware as 1 and goodware as -1
    Mal_labels = np.ones(len(AllMalSamples))
    Good_labels = np.empty(len(AllGoodSamples))
    Good_labels.fill(-1)
    y = np.concatenate((Mal_labels, Good_labels), axis=0)
    Logger.info("Label array - generated")

    # step 2: split all samples to training set and test set
    x_train_samplenames, x_test_samplenames, y_train, y_test = train_test_split(AllSampleNames, y, test_size=TestSize,
                                                     random_state=random.randint(0, 100), stratify=y)

    if(priorPortion!=0):
        x_train_samplenames, x_train_prior_samplenames, y_train, y_train_prior = train_test_split(x_train_samplenames, y_train, test_size=priorPortion,
                                                                                                  random_state=random.randint(255,65535), stratify=y_train)
        x_train_prior = FeatureVectorizer.transform(x_train_prior_samplenames)
    
    x_train = FeatureVectorizer.transform(x_train_samplenames)  
    x_test = FeatureVectorizer.transform(x_test_samplenames)
    Logger.debug("Test set split = %s", TestSize)
    Logger.info("train-test split done")
    if(saveTrainSet!=""):
        with open(os.path.join(saveTrainSet,f"trainSamples_{label}.pkl"),"wb") as f:
            pickle.dump((x_train_samplenames, y_train), f)
        if(priorPortion!=0):
            with open(os.path.join(saveTrainSet,f"priorSamples_{label}.pkl"),"wb") as f:
                pickle.dump((x_train_prior_samplenames, y_train_prior), f)
        
    
    # step 3: train the model
    Logger.info("Perform Classification with SVM Model")
    # Parameters= {'C': [0.01, 0.1, 1, 10, 100]}
    print(f"number of samples in training set: {x_train.shape}, number of samples in test set: {x_test.shape}")
    T0 = time.time() 
    if not Model:
        # Clf = GridSearchCV(LinearSVC(max_iter=1000000,dual=dual, penalty=penalty, fit_intercept=False), Parameters, cv= 5, scoring= 'f1', n_jobs=-1 )
        # SVMModels= Clf.fit(x_train, y_train)
        # Logger.info("Processing time to train and find best model with GridSearchCV is %s sec." %(round(time.time() -T0, 2)))
        # BestModel= SVMModels.best_estimator_
        def create_model(kernel):
            class_weights = {1: TestSize, 0: 1-TestSize}
            if kernel == 'linear':
                return LinearSVC(max_iter=1000000, C=1, dual=dual, fit_intercept=False, class_weight=class_weights)
            else:
                return SVC(kernel=kernel, max_iter=1000000, C=1, class_weight=class_weights)
        if(priorPortion!=0):
            PriorModel = create_model(kernel)
            PriorModel.fit(x_train_prior, y_train_prior)
            
        # BestModel = LinearSVC(max_iter=1000000,dual=dual, penalty=penalty, C=1, fit_intercept=False)
        BestModel = create_model(kernel)
        BestModel.fit(x_train, y_train)
        Logger.info("Best Model Selected : {}".format(BestModel))
    else:
        BestModel = joblib.load(Model)

    # step 4: Evaluate the best model on test set
    w = BestModel.coef_
    w_norm = w/norm(w)
    rounded = round(norm(w)/5)*5 
    if(rounded == 0):
        rounded = norm(w)
    step = rounded * 0.01
    mu_values = [rounded + step * i for i in range(-25, 26)]
    results = []
    full = 0
    for mu in mu_values:
        pacc,ptrain_acc,ptest_f1, ptrain_f1, ptest_loss, ptrain_loss = 0,0,0,0,0,0
        # test_f1, train_f1, acc, train_acc, test_loss, train_loss = error.evaluation_metrics(f"random classification with priorportion-{priorPortion}", BestModel, x_test, x_train, y_test, y_train)
        BestModel.coef_ = w_norm
        model = BestModel
        test_f1, train_f1, acc, train_acc, test_loss, train_loss = error.evaluation_metrics(f"random classification with normed priorportion-{priorPortion}", BestModel, x_test, x_train, y_test, y_train)
        if(priorPortion!=0):
            BestModel.coef_ = mu*w_norm
            ptest_f1, ptrain_f1, pacc, ptrain_acc, ptest_loss, ptrain_loss = error.evaluation_metrics("random classification using priorModel", BestModel, x_test, x_train, y_test, y_train)
            full = error.theory_specifics(f"random classification with priorportion-{priorPortion}", BestModel, mu=mu, prior=PriorModel, eta=eta)
            results.append([eta, num, mu, full, ptest_f1, ptrain_f1, pacc, ptrain_acc, ptest_loss, ptrain_loss])
        else:
            full = error.theory_specifics("random classification without prior", BestModel)
            results.append([eta, num, mu, full, test_f1, train_f1, acc, train_acc, test_loss, train_loss])

    return results, model, rounded
