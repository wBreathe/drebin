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
from sklearn.svm import LinearSVC, SVC
import error
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap


def umap_visualization(train_features, test_features, train_labels, test_labels, 
                       save_prefix='umap_visualization', subsample_size=5000, 
                       random_state=42, use_gpu=True):
    """
    UMAP可视化替代PCA版本
    Args:
        train/test_features: 稀疏矩阵格式特征
        train/test_labels: 0/1标签
        subsample_size: 子采样数量（避免内存不足）
        use_gpu: 是否使用RAPIDS GPU加速
    """
    # 合并数据并转换为密集数组
    all_features = vstack([train_features, test_features]).toarray()
    all_labels = np.concatenate([train_labels, test_labels])
    is_train = np.concatenate([np.ones(len(train_labels)), np.zeros(len(test_labels))]).astype(bool)

    # 子采样避免内存问题
    if len(all_features) > subsample_size:
        np.random.seed(random_state)
        idx = np.random.choice(len(all_features), subsample_size, replace=False)
        all_features = all_features[idx]
        all_labels = all_labels[idx]
        is_train = is_train[idx]

    # L2归一化提升UMAP效果
    all_features = normalize(all_features, norm='l2')

    # 初始化UMAP模型
    if use_gpu:
        from cuml.manifold import UMAP  # RAPIDS cuML GPU加速
        reducer = UMAP(
            n_components=2,
            n_neighbors=30,
            min_dist=0.1,
            random_state=random_state,
            verbose=False
        )
    else:
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=30,
            min_dist=0.1,
            random_state=random_state,
            metric='euclidean'
        )

    # 执行降维
    embedding = reducer.fit_transform(all_features)

    # 划分数据
    train_emb = embedding[is_train]
    test_emb = embedding[~is_train]
    train_labels_sub = all_labels[is_train]
    test_labels_sub = all_labels[~is_train]

    # 定义颜色和标签
    colors = {
        'train_goodware': ('blue', 'o'),
        'train_malware': ('red', 'o'),
        'test_goodware': ('cyan', '^'),
        'test_malware': ('orange', '^')
    }

    def plot_umap(data_dict, title, save_name):
        plt.figure(figsize=(10, 6))
        for key, (data, color, marker) in data_dict.items():
            plt.scatter(data[:, 0], data[:, 1], 
                        c=color, 
                        marker=marker,
                        label=key.replace('_', ' ').title(),
                        alpha=0.6, 
                        edgecolors='w',
                        linewidths=0.3,
                        s=30)
        plt.title(title)
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.legend(markerscale=1.5)
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_{save_name}.png', dpi=150)
        plt.close()

    # 1. 训练集内部对比
    train_good = train_emb[train_labels_sub == 0]
    train_mal = train_emb[train_labels_sub == 1]
    plot_umap(
        {
            'train_goodware': (train_good, colors['train_goodware'][0], 'o'),
            'train_malware': (train_mal, colors['train_malware'][0], 'o')
        },
        'UMAP: Train Data (Goodware vs Malware)',
        'train_internal'
    )

    # 2. 测试集内部对比
    test_good = test_emb[test_labels_sub == 0]
    test_mal = test_emb[test_labels_sub == 1]
    plot_umap(
        {
            'test_goodware': (test_good, colors['test_goodware'][0], '^'),
            'test_malware': (test_mal, colors['test_malware'][0], '^')
        },
        'UMAP: Test Data (Goodware vs Malware)',
        'test_internal'
    )

    # 3. Goodware跨域对比
    all_good = embedding[all_labels == 0]
    is_train_good = is_train[all_labels == 0]
    plot_umap(
        {
            'train_goodware': (all_good[is_train_good], colors['train_goodware'][0], 'o'),
            'test_goodware': (all_good[~is_train_good], colors['test_goodware'][0], '^')
        },
        'UMAP: Goodware (Train vs Test)',
        'goodware_cross_domain'
    )

    # 4. Malware跨域对比
    all_mal = embedding[all_labels == 1]
    is_train_mal = is_train[all_labels == 1]
    plot_umap(
        {
            'train_malware': (all_mal[is_train_mal], colors['train_malware'][0], 'o'),
            'test_malware': (all_mal[~is_train_mal], colors['test_malware'][0], '^')
        },
        'UMAP: Malware (Train vs Test)',
        'malware_cross_domain'
    )

    # 5. 全数据综合视图
    plot_umap(
        {
            'train_goodware': (train_good, colors['train_goodware'][0], 'o'),
            'train_malware': (train_mal, colors['train_malware'][0], 'o'),
            'test_goodware': (test_good, colors['test_goodware'][0], '^'),
            'test_malware': (test_mal, colors['test_malware'][0], '^')
        },
        'UMAP: All Data',
        'full_view'
    )


def pca_visualization(train_features, test_features, train_labels, test_labels, save_prefix='pca_visualization_linear_rbf'):
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

def getFeature(dir, years):
    MalwareCorpus=os.path.join(dir, "training", "malware")
    GoodwareCorpus=os.path.join(dir, "training", "goodware")
    AllMalSamples = CM.ListFiles(MalwareCorpus, ".data", year=years)
    AllGoodSamples = CM.ListFiles(GoodwareCorpus, ".data", year=years)
    return AllMalSamples, AllGoodSamples

def rbf_kernel_torch(X, Y, gamma=None):
    if gamma is None:
        gamma = 1.0 / X.shape[1] 
    X_norm = torch.sum(X ** 2, dim=1).view(-1, 1)
    Y_norm = torch.sum(Y ** 2, dim=1).view(1, -1)
    K = torch.exp(-gamma * (X_norm + Y_norm - 2 * torch.matmul(X, Y.T)))
    return K

def polynomial_kernel_torch(X, Y, degree=3, coef0=1, gamma=None):
    """PyTorch implementation of polynomial kernel."""
    if gamma is None:
        gamma = 0.01
    return (gamma * torch.matmul(X, Y.T) + coef0) ** degree

def compute_mmd(X_s, X_t, degree=3, coef0=1, gamma=None):
    """
    calculate mmd between source domain and target domain with a polynomial kernel
    """
    K_ss = rbf_kernel_torch(X_s, X_s, gamma=gamma)
    K_tt = rbf_kernel_torch(X_t, X_t, gamma=gamma)
    K_st = rbf_kernel_torch(X_s, X_t, gamma=gamma)
    print("K_ss: ", torch.min(K_ss), torch.max(K_ss))
    print("K_tt: ", torch.min(K_tt), torch.max(K_tt))
    print("K_st: ", torch.min(K_st), torch.max(K_st))
    mmd = torch.mean(K_ss) + torch.mean(K_tt) - 2 * torch.mean(K_st)
    mmd = torch.clamp(mmd, min=0.0)
    return mmd


def get_batch(sparse_matrix, device, batch_size=256):
    num_samples = sparse_matrix.shape[0]
    idx = 0

    while idx < num_samples:
        batch_end = min(idx + batch_size, num_samples)  
        batch = sparse_matrix[idx:batch_end].toarray() 
        idx = batch_end
        dense_batch = torch.tensor(batch, dtype=torch.float32, device=device)
        yield dense_batch



# class Autoencoder(nn.Module):
#     def __init__(self, input_dim, hidden_dim):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(),
#             nn.Linear(1024, hidden_dim),
#             nn.BatchNorm1d(hidden_dim)
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(hidden_dim, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, input_dim),
#             nn.Sigmoid()
#         )
#         for layer in self.modules():
#             if isinstance(layer, nn.Linear):
#                 nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')

#     def forward(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        encoded = torch.relu(self.encoder(x))  
        decoded = torch.sigmoid(self.decoder(encoded)) 
        return decoded



def extract_representations(model, features, batch_size=256, device="cuda"):
    model.eval()
    representations = []
    with torch.no_grad():
        for dense_batch in get_batch(features, device=device):
            encoded_batch = model.encoder(dense_batch)
            representations.append(encoded_batch)

    return torch.cat(representations, dim=0)

def main():
    dir = "/home/wang/Data/android"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    train_features = shuffle(train_features, random_state=2314)
    test_features = shuffle(test_features, random_state=2314)
    
    input_dim = train_features.shape[1]
    assert(input_dim==test_features.shape[1])
    hidden_dim = 512

    model = Autoencoder(input_dim, hidden_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    loss_function = nn.BCELoss()  


    num_epochs = 20 

    save_loss = 100000
    for epoch in range(num_epochs):
        model.train()  
        running_loss = 0.0
        running_mmd_loss = 0.0
        running_recon_loss_train = 0.0
        running_recon_loss_test = 0.0

        test_batch_cycle = cycle(get_batch(test_features, device))

        for batch_idx, dense_batch_train in enumerate(get_batch(train_features, device)):
            optimizer.zero_grad()

            dense_batch_test = next(test_batch_cycle)
            outputs_train = model(dense_batch_train)
            outputs_test = model(dense_batch_test)
            representation_train = model.encoder(dense_batch_train)
            representation_test = model.encoder(dense_batch_test)

            mmd_loss = compute_mmd(representation_train, representation_test)
            reconstruction_loss_train = loss_function(outputs_train, dense_batch_train)
            reconstruction_loss_test = loss_function(outputs_test, dense_batch_test)

            total_loss = reconstruction_loss_train + reconstruction_loss_test + mmd_loss
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            running_mmd_loss += mmd_loss.item()
            running_recon_loss_train += reconstruction_loss_train.item()
            running_recon_loss_test += reconstruction_loss_test.item()

            print(f"Batch {batch_idx + 1}: "
                f"MMD Loss: {mmd_loss.item():.4f}, "
                f"Recon Train Loss: {reconstruction_loss_train.item():.4f}, "
                f"Recon Test Loss: {reconstruction_loss_test.item():.4f}, "
                f"Total Loss: {total_loss.item():.4f}")
            
        print(f"Epoch [{epoch+1}/{num_epochs}], "
            f"Total MMD Loss: {running_mmd_loss:.4f}, "
            f"Total Recon Train Loss: {running_recon_loss_train:.4f}, "
            f"Total Recon Test Loss: {running_recon_loss_test:.4f}, "
            f"Total Total Loss: {running_loss:.4f}")
        if(running_loss < save_loss):
            save_loss = running_loss
            # torch.save(model.state_dict(), "/home/wang/Data/android/autoencoder_1_rbf_epoch-30—best_hidden-512_batch-256.pth")

    # bmodel = Autoencoder(input_dim, hidden_dim).to(device)
    # bmodel.load_state_dict(torch.load("/home/wang/Data/android/autoencoder_1_rbf_epoch-30—best_hidden-512_batch-256.pth")) 
    # bmodel.to(device)
    del train_features, test_features
    gc.collect()
    train_goodware = extract_representations(model, goodfeatures).detach().cpu().numpy()
    train_malware = extract_representations(model, malfeatures).detach().cpu().numpy()
    test_goodware = extract_representations(model, tgoodfeatures).detach().cpu().numpy()
    test_malware = extract_representations(model, tmalfeatures).detach().cpu().numpy()
    del goodfeatures, malfeatures, tgoodfeatures, tmalfeatures
    gc.collect()
    train_features = np.concatenate([train_goodware, train_malware], axis=0)
    test_features = np.concatenate([test_goodware, test_malware], axis=0)
    train_labels = np.concatenate([np.zeros(train_goodware.shape[0]), np.ones(train_malware.shape[0])])
    test_labels = np.concatenate([np.zeros(test_goodware.shape[0]), np.ones(test_malware.shape[0])])
    train_features, train_labels = shuffle(train_features, train_labels, random_state=1423)
    test_features, test_labels = shuffle(test_features, test_labels, random_state=1423)
    # svcModel = LinearSVC(max_iter=1000000, C=1, dual=False, fit_intercept=False)
    svcModel = SVC(kernel='rbf', C=1, gamma='scale')
    svcModel.fit(train_features, train_labels)


    test_f1, train_f1, acc, train_acc, test_loss, train_loss = error.evaluation_metrics(f"Support shift", svcModel, test_features, train_features, test_labels, train_labels)
    
    pca_visualization(train_features, test_features, train_labels, test_labels)
    # umap_visualization(
    #     train_features, 
    #     test_features, 
    #     train_labels, 
    #     test_labels,
    #     subsample_size=100000,  # 根据GPU显存调整
    #     use_gpu=True,          # 使用RAPIDS加速
    #     save_prefix='umap_result'
    # )
    
    
if __name__=="__main__":
    main()
