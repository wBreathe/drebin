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



def pca_visualization(train_features, test_features, train_labels, test_labels, save_prefix='pca_visualization_linear_rbf_reverse'):
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
    # print("K_ss: ", torch.min(K_ss), torch.max(K_ss))
    # print("K_tt: ", torch.min(K_tt), torch.max(K_tt))
    # print("K_st: ", torch.min(K_st), torch.max(K_st))
    mmd = torch.mean(K_ss) + torch.mean(K_tt) - 2 * torch.mean(K_st)
    mmd = torch.clamp(mmd, min=0.0)
    return mmd


def get_batch(sparse_matrix, labels = None, device='cpu', batch_size=256):
    num_samples = sparse_matrix.shape[0]
    idx = 0
    while idx < num_samples:
        batch_end = min(idx + batch_size, num_samples)  
        batch = sparse_matrix[idx:batch_end].toarray() 
        dense_batch = torch.tensor(batch, dtype=torch.float32, device=device)
        if(labels is not None):
            batch_labels = labels[idx:batch_end] 
            yield dense_batch, batch_labels
        else:
            yield dense_batch
        idx = batch_end
                   

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_dim),
            nn.Sigmoid()
        )
        self.classifier = nn.Linear(hidden_dim, 2)  
        
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x, train=True):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        if(train):
            class_logits = self.classifier(encoded)
            return decoded, class_logits
        else:
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
    
    train_labels = np.hstack([np.zeros(goodfeatures.shape[0]), np.ones(malfeatures.shape[0])])
    test_labels = np.hstack([np.zeros(tgoodfeatures.shape[0]), np.ones(tmalfeatures.shape[0])])
    train_features, train_labels = shuffle(train_features, train_labels, random_state=2314)
    test_features, test_labels = shuffle(test_features, test_labels, random_state=2314)
    
    svcModel = LinearSVC(max_iter=1000000, C=1, dual=False, fit_intercept=False)
    svcModel.fit(train_features, train_labels)

    # train_preds = svcModel.predict(train_features)
    # test_preds = svcModel.predict(test_features)
    test_f1, train_f1, acc, train_acc, test_loss, train_loss = error.evaluation_metrics(f"Without Support shift", svcModel, test_features, train_features, test_labels, train_labels)
    
    iterations = 20
    for iter in range(iterations):
        print(f"Iteration: {iter}")
        test_scores = svcModel.decision_function(test_features)
        best_thresh = np.percentile(np.abs(test_scores), 90)  
        print(f"Threshold for top 10% confident samples: {best_thresh}")
    
        high_confidence_idx = np.where(np.abs(test_scores) >= best_thresh)[0]
        pseudo_labels = (test_scores[high_confidence_idx] >= 0).astype(int)

        pseudo_features = test_features[high_confidence_idx]

        
        num_positive = np.sum(pseudo_labels)  
        num_negative = len(pseudo_labels) - num_positive
        total_selected = len(pseudo_labels)

        print(f"Total selected samples: {total_selected}")
        print(f"Positive class (1): {num_positive} ({num_positive / total_selected:.2%})")
        print(f"Negative class (0): {num_negative} ({num_negative / total_selected:.2%})")
        test_f1, train_f1, acc, train_acc, test_loss, train_loss = error.evaluation_metrics(f"initial", svcModel, pseudo_features, train_features, test_labels[high_confidence_idx], train_labels)
    
        new_train_features = vstack([train_features, pseudo_features])
        new_train_labels = np.concatenate([train_labels, pseudo_labels])
        
        new_train_labels = torch.tensor(new_train_labels, dtype=torch.long).to(device)
    
        input_dim = new_train_features.shape[1]
        assert(input_dim==test_features.shape[1])
        hidden_dim = 512

        model = Autoencoder(input_dim, hidden_dim).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

        loss_function = nn.BCELoss()  
        classification_loss_function = nn.CrossEntropyLoss()

        num_epochs = 20 

        save_loss = 100000
        for epoch in range(num_epochs):
            model.train()  
            running_loss = 0.0
            running_mmd_loss = 0.0
            running_recon_loss_train = 0.0
            running_recon_loss_test = 0.0
            running_classification_loss = 0.0

            test_batch_cycle = cycle(get_batch(test_features, device = device))

            for batch_idx, (dense_batch_train, batch_labels) in enumerate(get_batch(new_train_features, labels=new_train_labels, device=device)):
                optimizer.zero_grad()

                dense_batch_test = next(test_batch_cycle)
                outputs_train, outputs_class = model(dense_batch_train, train=True)
                outputs_test = model(dense_batch_test, train=False)
                representation_train = model.encoder(dense_batch_train)
                representation_test = model.encoder(dense_batch_test)

                mmd_loss = compute_mmd(representation_train, representation_test)
                reconstruction_loss_train = loss_function(outputs_train, dense_batch_train)
                reconstruction_loss_test = loss_function(outputs_test, dense_batch_test)
                classification_loss_train = classification_loss_function(outputs_class, batch_labels)
                total_loss = reconstruction_loss_train + reconstruction_loss_test + 5*mmd_loss + 10*classification_loss_train
                total_loss.backward()
                optimizer.step()

                running_loss += total_loss.item()
                running_mmd_loss += mmd_loss.item()
                running_recon_loss_train += reconstruction_loss_train.item()
                running_recon_loss_test += reconstruction_loss_test.item()
                running_classification_loss += classification_loss_train.item()

                print(f"Batch {batch_idx + 1}: "
                    f"MMD Loss: {mmd_loss.item():.4f}, "
                    f"Recon Train Loss: {reconstruction_loss_train.item():.4f}, "
                    f"Recon Test Loss: {reconstruction_loss_test.item():.4f}, "
                    f"Classification Train Loss: {classification_loss_train.item():.4f}, "
                    f"Total Loss: {total_loss.item():.4f}")
            
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                f"Total MMD Loss: {running_mmd_loss:.4f}, "
                f"Total Recon Train Loss: {running_recon_loss_train:.4f}, "
                f"Total Recon Test Loss: {running_recon_loss_test:.4f}, "
                f"Total Class Train Loss: {classification_loss_train:.4f}, "
                f"Total Total Loss: {running_loss:.4f}")
            if(running_loss < save_loss):
                save_loss = running_loss
                bmodel = model
            # torch.save(model.state_dict(), "/home/wang/Data/android/autoencoder_1_rbf_epoch-30—best_hidden-512_batch-256.pth")

        # bmodel = Autoencoder(input_dim, hidden_dim).to(device)
        # bmodel.load_state_dict(torch.load("/home/wang/Data/android/autoencoder_1_rbf_epoch-30—best_hidden-512_batch-256.pth")) 
        # bmodel.to(device)
        model = bmodel
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
    
        pca_visualization(train_features, test_features, train_labels, test_labels, save_prefix=f'pca_visualization_linear_rbf_iteration-{iter}')

    
    
if __name__=="__main__":
    main()
