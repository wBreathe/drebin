import os
import numpy as np
import time
from sklearn.metrics import accuracy_score, f1_score
from sklearn import metrics
from numpy.linalg import norm
from multiprocessing import Pool
from dataclasses import dataclass
from typing import Optional, List
import logging

logging.basicConfig(level=logging.INFO)
Logger = logging.getLogger('general.stdout')
Logger.setLevel("INFO")

@dataclass
class RandomConfig:
    NCpuCores: int
    priorPortion: float
    eta: float
    mu: float
    dual: bool
    penalty: str
    years: List[int]
    enable_imbalance: bool
    MalwareCorpus: str
    GoodwareCorpus: str
    TestSize: float
    FeatureOption: str
    Model: str
    NumTopFeats: int
    saveTrainSet: str = ""
    enableFuture: bool = False
    futureYears: Optional[List[int]] = None
    futureMalwareCorpus: Optional[str] = None
    futureGoodwareCorpus: Optional[str] = None

@dataclass
class HoldoutConfig:
    NCpuCores: int
    priorPortion: float
    eta: float
    mu: float
    dual: bool
    penalty: str
    years: list
    saveTrainSet: str
    enable_imbalance: bool
    TestMalSet: str
    TestGoodSet: str
    TestSize: float
    FeatureOption: str
    Model: str
    NumTopFeats: int

def sample_spherical_gaussian_from_w(w, num_samples):
    # w needs to be normalized
    w = w.ravel()
    norm_w = norm(w)
    if(norm_w):
        w = w/norm_w
    else:
        raise Exception("Error: the norm of w equals to zero!")
    # cov_matrix = np.eye(len(w))
    Logger.info("before sample")
    # w_samples = np.random.multivariate_normal(w, cov_matrix, size=num_samples)
    w_samples = np.random.normal(loc=w, scale=1.0, size=(num_samples, len(w)))
    Logger.info("sample finished")
    return w_samples


def zero_one_loss(y_true, y_pred):
    return np.mean(y_true != y_pred)


def get_loss(model, w_samples, x, y_true):
    losses = [zero_one_loss(y_true, model.predict(x, w_prime)) for w_prime in w_samples]
    avg_loss = np.mean(losses)
    std_loss = np.std(losses)
    return avg_loss, std_loss

def get_loss_multiprocessing(model, w_samples, x, y_true, num_processes=4):
    def compute_loss(w_prime):
        return zero_one_loss(y_true, model.predict(x, w_prime))
    
    with Pool(processes=num_processes) as pool:
        losses = pool.map(compute_loss, w_samples)
    
    avg_loss = np.mean(losses)
    std_loss = np.std(losses)
    return avg_loss, std_loss

def evaluation_metrics(label, model, x_test, x_train, y_test, y_train):
    Logger.info("Start evaluation ......")
    T0 = time.time()
    y_pred = model.predict(x_test)
    y_train_pred = model.predict(x_train)
    Logger.info((f"The testing time for {label} is %s sec." % (round(time.time() - T0,2))))
    Accuracy = f1_score(y_test, y_pred, average='binary')
    Logger.info(("Test Set F1 = {}".format(Accuracy)))
    Train_Accuracy = f1_score(y_train, y_train_pred, average='binary')
    Logger.info(("Train Set F1 = {}".format(Train_Accuracy)))
    Logger.info((metrics.classification_report(y_test,
                                         y_pred, labels=[1, -1],
                                         target_names=['Malware', 'Goodware'])))
    Report = "Test Set F1 = " + str(Accuracy) + "\n" + metrics.classification_report(y_test,
                                                                                     y_pred,
                                                                                     labels=[1, -1],
                                                                                     target_names=['Malware',
                                                                                                'Goodware'])
    Logger.info(Report)
    return Report


def theory_specifics(label, model, prior=None, eta=0, mu=1):
    # pointwise multiplication between weight and feature vect
    Logger.info(f"The specifics for theoretical bounds: {label}")
    Logger.info(f"iteration in sum: {model.n_iter_}")
    all_parameters = np.prod(model.coef_.shape)
    Logger.info(f"all parameters: {all_parameters}")
    
    w = model.coef_
    l1_norm = norm(w, ord=1)
    l2_norm = norm(w)
    Logger.info(f"C:{model.C}")
    Logger.info(f"weights: {w}")
    Logger.info(f"l1 norm:{l1_norm}, l2 norm:{l2_norm}")
    
    if(prior):
        wr = prior.coef_
        w = w.ravel()
        norm_w = norm(w)
        if(norm_w):
            w = w/norm_w
        wr = wr.ravel()
        norm_wr = norm(wr)
        if(norm_wr):
            wr = wr/norm_wr
        l2_norm = norm(eta*wr-mu*w)
        if wr.shape != w.shape:
            raise ValueError("The shapes of wr and w do not match!")
        Logger.info(f"eta:{eta}, mu:{mu}, ||eta*wr-mu*w||2: {l2_norm}")