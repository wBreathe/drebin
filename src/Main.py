from GetApkData import GetApkData
from RandomClassification import RandomClassification
from HoldoutClassification import HoldoutClassification
import psutil, argparse, logging
import os
import sys
logging.basicConfig(level=logging.INFO)
Logger = logging.getLogger('main.stdout')
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, List

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
    i = Args.i
    priorPortion = Args.priorPortion
    eta = Args.eta
    mu = Args.mu
    future = Args.future
    current_date = datetime.now().strftime("%Y-%m-%d")
    label = f"{i}_dual-{dual}_penalty-{penalty}_{current_date}"
    log_file = open(f"{label}.log", "w")
    sys.stdout = log_file
    
    
    if(apk):
        apk_paths = [os.path.join(dir,"training",'malware'), os.path.join(dir,"training",'goodware'),os.path.join(dir,"test",'malware'),os.path.join(dir,"test",'goodware')]
        GetApkData(NCpuCores, *apk_paths)


    randomConfig = RandomConfig(
        NCpuCores=NCpuCores,
        priorPortion=priorPortion,
        eta=eta,
        mu=mu,
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
        NCpuCores=NCpuCores,
        priorPortion=priorPortion,
        eta=eta,
        mu=mu,
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

    RandomClassification(randomConfig)
    HoldoutClassification(holdoutConfig)
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
    Args.add_argument("--priorPortion", type=float, default=0,
                      help="The portion of samples randomly extracted as prior, the default value 0 indicates no prior is considred.")
    Args.add_argument("--future", type=bool, default=True,
                      help="Whether use future set for feature vectorization or not")
    Args.add_argument("--eta", type=int, default=10,
                      help="the scaling for prior distribution")
    Args.add_argument("--mu", type=int, default=10,
                      help="the scaling for posterior distribution")
    Args.add_argument("--apk", type=bool, default=False, 
                      help= "Whether to process APKs or not")
    Args.add_argument("--i", type=int, default=0, 
                      help= "the i th experiment")
    
    return Args.parse_args()

if __name__ == "__main__":
    main(ParseArgs(), True)
