from GetApkData import GetApkData
from RandomClassification import RandomClassification
from HoldoutClassification import HoldoutClassification
import psutil, argparse, logging
import os
logging.basicConfig(level=logging.INFO)
Logger = logging.getLogger('main.stdout')

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
     #GetApkData(NCpuCores, os.path.join(dir,"training",'malware'), os.path.join(dir,"training",'goodware'),os.path.join(dir,"test",'malware'),os.path.join(dir,"test",'goodware'))
    # GetApkData(NCpuCores, os.path.join(dir,"test",'malware'),os.path.join(dir,"test",'goodware'))
    # RandomClassification(years, 
    # enable_imbalance, MalwareCorpus, GoodwareCorpus, 
    # TestSize, FeatureOption, Model, NumTopFeats, saveTrainSet=""):
    RandomClassification(train_years, True, os.path.join(dir, "training", "malware"), os.path.join(dir, "training", "goodware"), TestSize, FeatureOption, Model, NumFeatForExp, os.path.join(dir, "training"))
    # HoldoutClassification(years, saveTrainSet, enable_imbalance, 
    # TestMalSet, TestGoodSet, TestSize, FeatureOption, Model, NumTopFeats):
    HoldoutClassification(["2021", "2022"], os.path.join(dir, "training"), True, os.path.join(dir, "test", "malware"), os.path.join(dir, "test", "goodware"), TestSize, FeatureOption, Model, NumFeatForExp)

def ParseArgs():
    Args =  argparse.ArgumentParser(description="Classification of Android Applications")
    Args.add_argument("--datadir", default= "/home/wang/Data/android/",
                      help= "Absolute path to directory containing apks")
    Args.add_argument("--ncpucores", type= int, default= psutil.cpu_count(),
                      help= "Number of CPUs that will be used for processing")
    Args.add_argument("--testsize", type= float, default= 0.3,
                      help= "Size of the test set when split by Scikit Learn's Train Test Split module")
    Args.add_argument("--model",
                      help= "Absolute path to the saved model file(.pkl extension)")
    Args.add_argument("--numfeatforexp", type= int, default = 30,
                      help= "Number of top features to show for each test sample")
    return Args.parse_args()

if __name__ == "__main__":
    main(ParseArgs(), True)
