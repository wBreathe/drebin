from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from itertools import cycle
import os
import CommonModules as CM
from sklearn.feature_extraction.text import TfidfVectorizer as TF
import numpy as np
from scipy.sparse import vstack
import pickle
from sklearn.utils import shuffle
import gc
from sklearn.svm import LinearSVC
import error

def getFeature(dir, years):
    MalwareCorpus=os.path.join(dir, "training", "malware")
    GoodwareCorpus=os.path.join(dir, "training", "goodware")
    AllMalSamples = CM.ListFiles(MalwareCorpus, ".data", year=years)
    AllGoodSamples = CM.ListFiles(GoodwareCorpus, ".data", year=years)
    return AllMalSamples, AllGoodSamples

def pca_visualization(train_features, test_features, train_labels, test_labels, save_prefix='pca_visualization_original'):
    # 合并训练和测试特征
    all_features = vstack([train_features, test_features])
    all_labels = np.concatenate([train_labels, test_labels])
    
    # 进行 PCA 降维到 2D
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(all_features.toarray())
    
    # 拆分回训练和测试数据
    train_pca = pca_result[:train_features.shape[0]]
    test_pca = pca_result[train_features.shape[0]:]
    
    # 获取不同类别的索引
    train_goodware = train_pca[train_labels == 0]
    train_malware = train_pca[train_labels == 1]
    test_goodware = test_pca[test_labels == 0]
    test_malware = test_pca[test_labels == 1]
    
    # 颜色定义
    colors = {
        'train_goodware': 'blue',
        'train_malware': 'red',
        'test_goodware': 'cyan',
        'test_malware': 'orange'
    }
    
    def plot_pca(data_dict, title, save_name):
        plt.figure(figsize=(10, 6))
        for label, (data, color) in data_dict.items():
            plt.scatter(data[:, 0], data[:, 1], c=color, label=label, alpha=0.6, edgecolors='k')
        plt.title(title)
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.legend()
        plt.savefig(f'{save_prefix}_{save_name}.png')
        plt.close()
    
    # 1. Train only (goodware & malware)
    plot_pca({
        'Train Goodware': (train_goodware, colors['train_goodware']),
        'Train Malware': (train_malware, colors['train_malware'])
    }, 'PCA: Train Data (Goodware & Malware)', 'train')
    
    # 2. Test only (goodware & malware)
    plot_pca({
        'Test Goodware': (test_goodware, colors['test_goodware']),
        'Test Malware': (test_malware, colors['test_malware'])
    }, 'PCA: Test Data (Goodware & Malware)', 'test')
    
    # 3. Goodware comparison (train vs test)
    plot_pca({
        'Train Goodware': (train_goodware, colors['train_goodware']),
        'Test Goodware': (test_goodware, colors['test_goodware'])
    }, 'PCA: Goodware (Train vs Test)', 'goodware')
    
    # 4. Malware comparison (train vs test)
    plot_pca({
        'Train Malware': (train_malware, colors['train_malware']),
        'Test Malware': (test_malware, colors['test_malware'])
    }, 'PCA: Malware (Train vs Test)', 'malware')
    
    # 5. All together
    plot_pca({
        'Train Goodware': (train_goodware, colors['train_goodware']),
        'Train Malware': (train_malware, colors['train_malware']),
        'Test Goodware': (test_goodware, colors['test_goodware']),
        'Test Malware': (test_malware, colors['test_malware'])
    }, 'PCA: All Data', 'all')


dir = "/home/wang/Data/android"
train_years = [str(i) for i in range(2014, 2020)]
test_years = [str(i) for i in range(2020, 2024)]
malwares, goodwares = getFeature(dir, train_years)
tmalwares, tgoodwares = getFeature(dir, test_years)
NewFeatureVectorizer = TF(input='filename', tokenizer=lambda x: x.split('\n'), token_pattern=None,binary=True)
NewFeatureVectorizer.fit(malwares+goodwares+tmalwares+tgoodwares)
goodfeatures = NewFeatureVectorizer.transform(goodwares)
malfeatures = NewFeatureVectorizer.transform(malwares)
tgoodfeatures = NewFeatureVectorizer.transform(tgoodwares)
tmalfeatures = NewFeatureVectorizer.transform(tmalwares)


train_features = vstack([goodfeatures, malfeatures])
test_features = vstack([tgoodfeatures, tmalfeatures])
train_labels = np.concatenate([np.zeros(goodfeatures.shape[0]), np.ones(malfeatures.shape[0])])
test_labels = np.concatenate([np.zeros(tgoodfeatures.shape[0]), np.ones(tmalfeatures.shape[0])])
# 调用函数并保存PCA图像
pca_visualization(train_features, test_features, train_labels, test_labels)
