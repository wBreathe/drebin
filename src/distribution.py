import torch
import torch.nn as nn
import os
import CommonModules as CM
from sklearn.feature_extraction.text import TfidfVectorizer as TF
import numpy as np
from sklearn.covariance import EmpiricalCovariance
import argparse
from sklearn.mixture import GaussianMixture
import pickle

def getFeature(dir, years):
    MalwareCorpus=os.path.join(dir, "training", "malware")
    GoodwareCorpus=os.path.join(dir, "training", "goodware")
    AllMalSamples = CM.ListFiles(MalwareCorpus, ".data", year=years)
    AllGoodSamples = CM.ListFiles(GoodwareCorpus, ".data", year=years)
    return AllMalSamples, AllGoodSamples


def get_batch(sparse_matrix, device, batch_size=256):
    num_samples = sparse_matrix.shape[0]
    idx = 0
    while idx < num_samples:
        batch_end = min(idx + batch_size, num_samples)  
        batch = sparse_matrix[idx:batch_end].toarray() 
        dense_batch = torch.tensor(batch, dtype=torch.float32, device=device)
        yield dense_batch
        idx = batch_end


class EncoderBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim,1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
    
    def forward(self, x):
        return self.encoder(x)

class DecoderBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_dim),
            nn.Sigmoid()
        )
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
    
    def forward(self, x):
        return self.decoder(x)

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = EncoderBlock(input_dim=input_dim, hidden_dim=hidden_dim)
        self.decoder = DecoderBlock(input_dim=input_dim, hidden_dim=hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 2)  
        nn.init.kaiming_normal_(self.classifier.weight, mode='fan_in', nonlinearity='relu')
       

    def forward(self, x):
        x_encoded = self.encoder(x)
        x_decoded = self.decoder(x_encoded)
      
        return x_encoded, x_decoded


def extract_representations(model, features, device):
    model.eval()
    representations = []
    with torch.no_grad():
        for dense_batch in get_batch(features, device=device):
            encoded_batch = model.encoder(dense_batch)
            representations.append(encoded_batch)

    return torch.cat(representations, dim=0)

def main():
    flags = ["train_goodware", "train_malware", "test_goodware", "test_malware"]
    Args =  argparse.ArgumentParser(description="Check the distribution")

    Args.add_argument("--flag", dest="flag", choices=flags,required=True)
    args = Args.parse_args()
    
    dir = "/home/wang/Data/android"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    train_years = [str(i) for i in range(2014, 2020)]
    test_years = [str(i) for i in range(2020, 2024)]
    

    malwares, goodwares = getFeature(dir, train_years)
    tmalwares, tgoodwares = getFeature(dir, test_years)
    NewFeatureVectorizer = TF(input='filename', tokenizer=lambda x: x.split('\n'), token_pattern=None, binary=True)
    NewFeatureVectorizer.fit(malwares+goodwares+tmalwares+tgoodwares)
    
    if(args.flag=="train_goodware"):
        features = NewFeatureVectorizer.transform(goodwares)
    elif(args.flag=="train_malware"):
        features = NewFeatureVectorizer.transform(malwares)
    elif(args.flag=="test_goodware"):
        features = NewFeatureVectorizer.transform(tgoodwares)
    elif(args.flag=="test_malware"):
        features = NewFeatureVectorizer.transform(tmalwares)
    
    input_dim = features.shape[1]
    hidden_dim = 512

    model = AutoEncoder(input_dim, hidden_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    loss_function = nn.BCELoss()

    num_epochs = 20 

    save_loss = 100000
    

    for epoch in range(num_epochs):
        model.train()  
        running_loss = 0.0

        for batch_idx, dense_batch_train in enumerate(get_batch(features, device=device)):
            optimizer.zero_grad()
            encoded, decoded = model(dense_batch_train)
            total_loss = loss_function(decoded, dense_batch_train)
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

            print(f"Batch {batch_idx + 1}: "
                f"Total Loss: {total_loss.item():.4f}")
        
        print(f"Epoch {epoch}: Train Encoded Mean={encoded.mean().item()}, Std={encoded.std().item()}")
        print(f"Epoch [{epoch+1}/{num_epochs}], "
            f"Total Total Loss: {running_loss:.4f}")
        if(running_loss < save_loss):
            save_loss = running_loss

    
    embeddings = extract_representations(model, features, device=device).detach().cpu().numpy()
    
    mean_vec = np.mean(embeddings, axis=0)
    cov_matrix = EmpiricalCovariance().fit(embeddings).covariance_

    print("Mean:", mean_vec)
    print("Covariance Matrix:", cov_matrix)
    
    gmm = GaussianMixture(n_components=1, covariance_type='full', random_state=42)
    gmm.fit(embeddings)

    mean_vec = gmm.means_[0]
    cov_matrix = gmm.covariances_[0]

    print("Estimated Mean:", mean_vec)
    print("Estimated Covariance Matrix:", cov_matrix)
    
    gmm_params = {
        "mean": mean_vec,
        "covariance": cov_matrix
    }

    
    with open(os.path.join(dir, f"{args.flag}_gmm.pkl"), "wb") as f:
        pickle.dump(gmm_params, f)
    
    # tsne_visualization(train_features, test_features, train_labels, test_labels)

    log_likelihood = gmm.score_samples(embeddings)
    mean_ll = np.mean(log_likelihood)
    std_ll = np.std(log_likelihood)

    threshold = mean_ll - 2 * std_ll

    anomalous_indices = np.where(log_likelihood < threshold)[0]
    num_anomalies = len(anomalous_indices)

    print(f"Total Samples: {len(log_likelihood)}")
    print(f"Anomalous Samples (log_likelihood < {threshold:.2f}): {num_anomalies}")
    print("Anomalous Sample Indices:", anomalous_indices)
    print("Mean of log_likelihood: ", mean_ll)
    print(log_likelihood[anomalous_indices])

if __name__=="__main__":
    main()
