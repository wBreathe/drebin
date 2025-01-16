from GetApkData import GetApkData
from RandomClassification import RandomClassification
from HoldoutClassification import HoldoutClassification
import psutil, argparse, logging
import os
import sys
logging.basicConfig(level=logging.INFO)
Logger = logging.getLogger('main.stdout')
from datetime import datetime
from error import RandomConfig, HoldoutConfig
import numpy as np
import pickle
from collections import defaultdict
import time

def main(Args, FeatureOption):
    '''
    Main function for malware and goodware classification
    :param args: arguments acquired from command lines(refer to ParseArgs() for list of args)
    :param FeatureOption: False
    '''

    dir= Args.datadir
    NCpuCores= Args.ncpucores
    Model= Args.model
    NumFeatForExp = Args.numfeatforexp
    train_years = ["2018", "2019", "2020"]
    TestSize= Args.testsize
    dual = Args.dual
    penalty = Args.penalty
    apk = Args.apk
    num = Args.num
    priorPortion = Args.priorPortion
    # eta = Args.eta
    # mu = Args.mu
    kernel = Args.kernel
    future = True if(Args.future!=0) else False
    current_date = datetime.now().strftime("%Y-%m-%d")
    label = f"{num}_dual-{dual}_penalty-{penalty}_priorPortion-{priorPortion}_future-{future}_{current_date}"
    log_file = open(f"{label}.log", "w")
    sys.stdout = log_file
    
    
    if(apk):
        apk_paths = [os.path.join(dir,"training",'malware'), os.path.join(dir,"training",'goodware'),os.path.join(dir,"test",'malware'),os.path.join(dir,"test",'goodware')]
        GetApkData(NCpuCores, *apk_paths)

    etas = [i for i in range(1, 100, 5)]
    random_results, holdout_results = [], []
    for eta in etas:
        for index in range(num):
            randomConfig = RandomConfig(
                kernel=kernel,
                NCpuCores=NCpuCores,
                priorPortion=priorPortion,
                eta=eta,
                dual=dual,
                penalty=penalty,
                years=train_years,
                enable_imbalance=True,
                MalwareCorpus=os.path.join(dir, "training", "malware"),
                GoodwareCorpus=os.path.join(dir, "training", "goodware"),
                TestSize=TestSize,
                FeatureOption=FeatureOption,
                Model=Model,
                NumTopFeats=NumFeatForExp,
                saveTrainSet=os.path.join(dir, "training"),
                enableFuture=future,
                futureYears=["2021", "2022"],
                futureMalwareCorpus=os.path.join(dir, "test", "malware"),
                futureGoodwareCorpus=os.path.join(dir, "test", "goodware")
            )
            holdoutConfig = HoldoutConfig(
                kernel=kernel,
                NCpuCores=NCpuCores,
                priorPortion=priorPortion,
                eta=eta,
                dual=dual,
                penalty=penalty,
                years=["2021", "2022"],
                saveTrainSet=os.path.join(dir, "training"),
                enable_imbalance=True,
                TestMalSet=os.path.join(dir, "test", "malware"),
                TestGoodSet=os.path.join(dir, "test", "goodware"),
                TestSize=TestSize,
                FeatureOption=FeatureOption,
                Model=Model,
                NumTopFeats=NumFeatForExp
            )
            temp_results_random, model, rounded = RandomClassification(index, randomConfig)
            temp_results_holdout = HoldoutClassification(index, model, rounded, holdoutConfig)
            random_results.extend(temp_results_random)
            holdout_results.extend(temp_results_holdout)


    stats = defaultdict(lambda: defaultdict(list))
    results = defaultdict(dict)
    # full, test_f1, future_test_f1, test_acc, future_test_acc, test_loss, future_test_loss, train_loss 

    for eta, num, mu, f, t_f, train_f1, t_a, train_acc, t_l, train_loss in random_results:
        key = (eta, mu)            
        stats["full"][key].append(f)
        stats["test_f1"][key].append(t_f)
        stats["test_acc"][key].append(t_a)
        stats["test_loss"][key].append(t_l)
        stats["train_loss"][key].append(train_loss)

    for eta, i, mu, ptest_f1, ptrain_f1, pacc, ptrain_acc, ptest_loss, ptrain_loss in holdout_results:
        key = (eta, mu)
        stats['future_test_f1'][key].append(ptest_f1)
        stats["future_test_acc"][key].append(pacc)
        stats["future_test_loss"][key].append(ptest_loss)

    for metric, value_dict in stats.items():
        for key, values in value_dict.items():
            avg = np.mean(np.array(values))
            std = np.std(np.array(values))
            results[key][metric] = (avg, std)

    for key, metrics in results.items():
        print(f"Results for {key}:")
        for metric, (avg, std) in metrics.items():
            print(f"  {metric}: Mean = {avg:.4f}, Std = {std:.4f}")
    
    with open(os.path.join(dir, f"{kernel}_stats_{int(time.time())}"), "wb") as f:
        pickle.dump(results, f)
    with open(os.path.join(dir, f"{kernel}_results_of_random_{int(time.time())}"),"wb") as f:
        pickle.dump(random_results, f)
    with open(os.path.join(dir, f"{kernel}_results_of_holdout_{int(time.time())}"),"wb") as f:
        pickle.dump(holdout_results, f)
    log_file.close()



def ParseArgs():
    Args =  argparse.ArgumentParser(description="Classification of Android Applications")

    Args.add_argument("--datadir", default= "/home/wang/Data/android/",
                      help= "Absolute path to directory containing apks")
    Args.add_argument("--ncpucores", type= int, default= 8,
                      help= "Number of CPUs that will be used for processing")
    Args.add_argument("--testsize", type= float, default= 0.3,
                      help= "Size of the test set when split by Scikit Learn's Train Test Split module")
    Args.add_argument("--model",
                      help= "Absolute path to the saved model file(.pkl extension)")
    Args.add_argument("--numfeatforexp", type= int, default = 30,
                      help= "Number of top features to show for each test sample")
    Args.add_argument("--penalty", type=str, default="l2", 
                      help="the penalty during training")
    Args.add_argument("--dual", type=bool, default=False, 
                      help="Whether use dual optimization for svm or not")
    Args.add_argument("--priorPortion", type=float, default=0.2,
                      help="The portion of samples randomly extracted as prior, the default value 0 indicates no prior is considred.")
    Args.add_argument("--future", type=int, default=1,
                      help="Whether use future set for feature vectorization or not")
    # Args.add_argument("--eta", type=int, default=10,
    #                   help="the scaling for prior distribution")
    # Args.add_argument("--mu", type=int, default=10,
    #                   help="the scaling for posterior distribution")
    Args.add_argument("--apk", type=bool, default=False, 
                      help= "Whether to process APKs or not")
    Args.add_argument("--num", type=int, default=100, 
                      help= "the i th experiment")
    Args.add_argument('--kernel', type=str, choices=['linear', 'poly', 'rbf', 'sigmoid'], default="linear", help='Kernel type (linear, poly, rbf, sigmoid)')
    return Args.parse_args()

if __name__ == "__main__":
    main(ParseArgs(), True)
