import torch
import torch.nn as nn
from itertools import cycle
import os
import CommonModules as CM
from sklearn.feature_extraction.text import TfidfVectorizer as TF
import numpy as np
from scipy.sparse import vstack
import pickle


def getFeature(dir, years):
    MalwareCorpus=os.path.join(dir, "training", "malware")
    GoodwareCorpus=os.path.join(dir, "training", "goodware")
    AllMalSamples = CM.ListFiles(MalwareCorpus, ".data", year=years)
    AllGoodSamples = CM.ListFiles(GoodwareCorpus, ".data", year=years)
    return AllMalSamples, AllGoodSamples


def sparse_to_dense(sparse_mtx, device):
    return sparse_mtx.to_dense().to(device)


def polynomial_kernel_torch(X, Y, degree=3, coef0=1, gamma=None):
    """PyTorch implementation of polynomial kernel."""
    if gamma is None:
        gamma = 1.0 / X.shape[1]  
    return (gamma * torch.matmul(X, Y.T) + coef0) ** degree

def compute_mmd(X_s, X_t, degree=3, coef0=1, gamma=None):
    """
    calculate mmd between source domain and target domain with a polynomial kernel
    """
    K_ss = polynomial_kernel_torch(X_s, X_s, degree=degree, coef0=coef0, gamma=gamma)
    K_tt = polynomial_kernel_torch(X_t, X_t, degree=degree, coef0=coef0, gamma=gamma)
    K_st = polynomial_kernel_torch(X_s, X_t, degree=degree, coef0=coef0, gamma=gamma)
    
    mmd = torch.mean(K_ss) + torch.mean(K_tt) - 2 * torch.mean(K_st)
    return mmd


def sparse_to_torch_coo(sparse_mtx):
    coo = sparse_mtx.tocoo() 
    indices = torch.LongTensor(np.vstack((coo.row, coo.col)))
    values = torch.FloatTensor(coo.data) 
    shape = torch.Size(coo.shape)
    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)


def get_batch(sparse_matrix, device, batch_size=32):
    num_samples = sparse_matrix.shape[0]
    idx = 0

    while idx < num_samples:
        batch_end = min(idx + batch_size, num_samples)  
        batch = sparse_matrix[idx:batch_end] 
        idx = batch_end
        dense_batch = torch.cat([sparse_to_dense(sparse_to_torch_coo(sparse_mtx), device) for sparse_mtx in batch], dim=0)
        yield dense_batch



class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        encoded = torch.relu(self.encoder(x))  
        decoded = torch.sigmoid(self.decoder(encoded)) 
        return decoded


def extract_representations(model, features, batch_size=32, device="cuda"):
    model.eval()
    representations = []

    for dense_batch in get_batch(features, device=device):
        encoded_batch = model.encoder(dense_batch)
        representations.append(encoded_batch)

    return torch.cat(representations, dim=0)

def main():
    dir = "/home/wang/Data/android"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_years = [str(i) for i in range(2016,2020)]
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
    
    input_dim = train_features.shape[1]
    assert(input_dim==test_features.shape[1])
    hidden_dim = 128

    model = Autoencoder(input_dim, hidden_dim).to(device)

    #'''
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.BCELoss()  


    num_epochs = 50  


    for epoch in range(num_epochs):
        model.train()  
        running_loss = 0.0
    
        test_batch_cycle = cycle(get_batch(test_features, device))

        for dense_batch_train in get_batch(train_features, device):
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
    
        avg_loss = running_loss / (train_features.shape[0])
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    torch.save(model.state_dict(), "/home/wang/Data/android/autoencoder.pth")
    #'''
    model.load_state_dict(torch.load("/home/wang/Data/android/autoencoder.pth")) 
    model.to(device)
    
    train_vectors = extract_representations(model, train_features)
    test_vectors = extract_representations(model, test_features)
    
    with open("/home/wang/Data/android/train_vectors.pkl", "wb") as f:
        pickle.dump(train_vectors, f)
    
    with open("/home/wang/Data/android/test_vectors.pkl", "wb") as f:
        pickle.dump(test_vectors, f)

if __name__=="__main__":
    main()