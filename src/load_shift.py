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


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        encoded = torch.relu(self.encoder(x))  
        decoded = torch.sigmoid(self.decoder(encoded)) 
        return decoded

def sparse_to_dense(sparse_mtx, device):
    return sparse_mtx.to_dense().to(device)

def sparse_to_torch_coo(sparse_mtx):
    coo = sparse_mtx.tocoo() 
    indices = torch.LongTensor(np.vstack((coo.row, coo.col)))
    values = torch.FloatTensor(coo.data) 
    shape = torch.Size(coo.shape)
    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)

def get_batch(sparse_matrix, device, batch_size=16):
    num_samples = sparse_matrix.shape[0]
    idx = 0

    while idx < num_samples:
        batch_end = min(idx + batch_size, num_samples)  
        batch = sparse_matrix[idx:batch_end].toarray() 
        idx = batch_end
        dense_batch = torch.tensor(batch, dtype=torch.float32, device=device)
        yield dense_batch

def extract_representations(model, features, batch_size=32, device="cuda"):
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
    input_dim = train_features.shape[1]
    hidden_dim = 512
    model = Autoencoder(input_dim, hidden_dim).to(device)
    model.load_state_dict(torch.load("/home/wang/Data/android/autoencoder_20_512.pth")) 
    model.to(device)
    
    # train_vectors = extract_representations(model, train_features)
    # test_vectors = extract_representations(model, test_features)
    
    # with open("/home/wang/Data/android/train_vectors.pkl", "wb") as f:
    #     pickle.dump(train_vectors, f)
    
    # with open("/home/wang/Data/android/test_vectors.pkl", "wb") as f:
    #     pickle.dump(test_vectors, f)
    del train_features
    gc.collect()
    train_goodware = extract_representations(model, goodfeatures).detach().cpu().numpy()
    print(train_goodware.shape)
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
    svcModel = LinearSVC(max_iter=1000000, C=1, dual=False, fit_intercept=False)
    svcModel.fit(train_features, train_labels)

    # train_preds = svcModel.predict(train_features)
    # test_preds = svcModel.predict(test_features)
    test_f1, train_f1, acc, train_acc, test_loss, train_loss = error.evaluation_metrics(f"Support shift", svcModel, test_features, train_features, test_labels, train_labels)
    

    
    


if __name__ == "__main__":
    main()
