import os
import numpy as np
import time
from sklearn.metrics import f1_score, accuracy_score
from sklearn import metrics
from numpy.linalg import norm
from dataclasses import dataclass
from typing import Optional, List
import logging
from joblib import Parallel, delayed

logging.basicConfig(level=logging.INFO)
Logger = logging.getLogger('general.stdout')
Logger.setLevel("INFO")

@dataclass
class RandomConfig:
    kernel: str
    NCpuCores: int
    priorPortion: float
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
    enableSample: bool = False
    futureYears: Optional[List[int]] = None
    futureMalwareCorpus: Optional[str] = None
    futureGoodwareCorpus: Optional[str] = None

@dataclass
class HoldoutConfig:
    kernel: str
    NCpuCores: int
    priorPortion: float
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
    enableSample: bool=False

def sample_spherical_gaussian_from_w(mu, w, num_samples):
    # w needs to be normalized
    w = w.ravel()
    norm_w = norm(w)
    if(norm_w):
        w = mu*w/norm_w
    else:
        raise Exception("Error: the norm of w equals to zero!")
    # cov_matrix = np.eye(len(w))
    # w_samples = np.random.multivariate_normal(w, cov_matrix, size=num_samples)
    w_samples = np.random.normal(loc=w, scale=1.0, size=(num_samples, len(w)))
    return w_samples


# def zero_one_loss(y_true, y_pred):
#     return np.mean(y_true != y_pred), f1_score(y_true, y_pred, average='binary')



# def get_loss_multiprocessing(model, w_samples, x, y_true, num_processes=4):
#     def compute_loss(w_prime):
#         model.coef_ = w_prime.reshape(1,-1)
#         y_pred = model.predict(x)
#         return zero_one_loss(y_true, y_pred)
    
#     results = Parallel(n_jobs=num_processes)(
#         delayed(compute_loss)(w_prime) for w_prime in w_samples
#     )
#     losses, f1_scores = zip(*results)
#     print('losses', losses)
#     print('f1s', f1_scores)
#     avg_loss = np.mean(losses)
#     std_loss = np.std(losses)
#     return avg_loss, std_loss


def evaluation_metrics(label, model, x_test, x_train, y_test, y_train):
    print("Start evaluation ......")
    T0 = time.time()
    y_pred = model.predict(x_test)
    y_train_pred = model.predict(x_train)
    print((f"The testing time for {label} is %s sec." % (round(time.time() - T0,2))))
    f1 = f1_score(y_test, y_pred, average='binary')
    print(("Test Set F1 = {}".format(f1)))
    Train_f1 = f1_score(y_train, y_train_pred, average='binary')
    print(("Train Set F1 = {}".format(Train_f1)))
    Acc = accuracy_score(y_test, y_pred)
    print(("Test Set acc = {}".format(Acc)))
    Train_Acc = accuracy_score(y_train, y_train_pred)
    print(("Train Set acc = {}".format(Train_Acc)))
    train_loss = np.mean(y_train != y_train_pred)
    test_loss = np.mean(y_test != y_pred)
    print("Train set zero-one-loss: ", test_loss)
    print("Test set zero-one-loss: ", train_loss)
    print((metrics.classification_report(y_test,
                                         y_pred, labels=[1, -1],
                                         target_names=['Malware', 'Goodware'])))
    Report = "Test Set F1 = " + str(f1) + "\n" + metrics.classification_report(y_test,
                                                                                     y_pred,
                                                                                     labels=[1, -1],
                                                                                     target_names=['Malware',
                                                                                                'Goodware'])
    print(Report)
    return f1, Train_f1, Acc, Train_Acc, test_loss, train_loss


def theory_specifics(label, model, mu=1, prior=None):
    # pointwise multiplication between weight and feature vect
    print(f"The specifics for theoretical bounds: {label}")
    print(f"iteration in sum: {model.n_iter_}")
    # all_parameters = np.prod(model.coef_.shape)
    # print(f"all parameters: {all_parameters}")
    
    w = model.coef_
    # l1_norm = norm(w, ord=1)
    # l2_norm = norm(w)
    # print(f"C:{model.C}")
    # print(f"weights: {w}")
    # print(f"l1 norm:{l1_norm}, l2 norm:{l2_norm}")
    def get_eta(w1, w2):
        w1 = w1 / np.linalg.norm(w1)
        w2 = w2 / np.linalg.norm(w2)
        cos_theta = np.dot(w1, w2)
        return cos_theta
        
    full = 0
    if(prior):
        wr = prior.coef_
        w = w.ravel()
        wr = wr.ravel()
        norm_wr = norm(wr)
        if(norm_wr):
            wr = wr/norm_wr
        eta = get_eta(wr, w)
        full = norm(eta*wr-w)
        if wr.shape != w.shape:
            raise ValueError("The shapes of wr and w do not match!")
        print("wr-w", wr-w)
        print(norm(wr-w))
        print(f"eta:{eta}, mu:{mu}, ||eta*wr-mu*w||2: {full}")
    return full
    
