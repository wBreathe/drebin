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
from error import RandomConfig


logging.basicConfig(level=logging.INFO)
Logger = logging.getLogger('RandomClf.stdout')
Logger.setLevel("INFO")


def RandomClassification(i:int, config: RandomConfig):
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
    NCpuCores = config.NCpuCores
    priorPortion = config.priorPortion
    eta = config.eta
    mu = config.mu
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
    label = f"_{i}_dual-{dual}_penalty-{penalty}_priorPortion-{priorPortion}"
    AllMalSamples = CM.ListFiles(MalwareCorpus, ".data", year=years)
    AllGoodSamples = CM.ListFiles(GoodwareCorpus, ".data", year=years)


    FeatureVectorizer = TF(input='filename', tokenizer=lambda x: x.split('\n'), token_pattern=None,
                           binary=FeatureOption)
    if(enableFuture):
        assert((futureYears is not None) and (futureMalwareCorpus is not None) and (futureGoodwareCorpus is not None))
        FutureMalSamples = CM.ListFiles(futureMalwareCorpus, ".data", year=futureYears)
        FutureGoodSamples = CM.ListFiles(futureGoodwareCorpus, ".data", year=futureYears)
        FeatureVectorizer.fit(AllMalSamples + AllGoodSamples + FutureGoodSamples + FutureMalSamples)
        with open(os.path.join(saveTrainSet, f"featureVector_{label}.pkl"), "wb") as f:
            pickle.dump(AllMalSamples+AllGoodSamples+FutureGoodSamples+FutureMalSamples, f)
    
    if(enable_imbalance):
        AllMalSamples = random.sample(AllMalSamples, int(0.2*len(AllGoodSamples))) if len(AllMalSamples)>int(0.2*len(AllGoodSamples)) else AllMalSamples
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
        if(priorPortion!=0):
            PriorModel = LinearSVC(max_iter=1000000, dual=dual, penalty=penalty, C=1, fit_intercept=False)
            PriorModel.fit(x_train_prior, y_train_prior)
            
        BestModel = LinearSVC(max_iter=1000000,dual=dual, penalty=penalty, C=1, fit_intercept=False)
        BestModel.fit(x_train, y_train)
        Logger.info("Best Model Selected : {}".format(BestModel))
    else:
        BestModel = joblib.load(Model)

    # step 4: Evaluate the best model on test set
    w = BestModel.coef_
    w_norm = w/norm(w)
    ptest_f1, ptrain_f1, ptest_loss, ptrain_loss = 0,0,0,0
    test_f1, train_f1, acc, train_acc, test_loss, train_loss = error.evaluation_metrics(f"random classification with priorportion-{priorPortion}", BestModel, x_test, x_train, y_test, y_train)
    BestModel.coef_ = w_norm
    test_f1, train_f1, acc, train_acc, test_loss, train_loss = error.evaluation_metrics(f"random classification with normed priorportion-{priorPortion}", BestModel, x_test, x_train, y_test, y_train)
    if(priorPortion!=0):
        ptest_f1, ptrain_f1, pacc, ptrain_acc, ptest_loss, ptrain_loss = error.evaluation_metrics("random classification using priorModel", PriorModel, x_test, x_train, y_test, y_train)
        l1_norm, l2_norm, full = error.theory_specifics(f"random classification with priorportion-{priorPortion}", BestModel, prior=PriorModel, eta=eta, mu=mu)
    else:
        l1_norm, l2_norm, full = error.theory_specifics("random classification without prior", BestModel)

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

    return {'f1_test':test_f1, 'f1_train':train_f1, 'acc_test':acc, 'acc_train':train_acc,'loss_test':test_loss, "loss_train":train_loss, "f1_test_prior":ptest_f1, "f1_train_prior": ptrain_f1, "acc_test_prior":pacc, "acc_train_prior":ptrain_acc, "loss_test_prior":ptest_loss, "loss_train_prior":ptrain_loss, "l1_norm":l1_norm, "l2_norm":l2_norm, "full":full}
