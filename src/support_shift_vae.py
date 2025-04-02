from argparse import ArgumentParser
import torch
import torch.nn as nn
from itertools import cycle
import os
import CommonModules as CM
from sklearn.feature_extraction.text import TfidfVectorizer as TF
import numpy as np
from scipy.sparse import vstack
from sklearn.utils import shuffle
import gc
from sklearn.svm import SVC
import error
from vanilla_vae import VanillaVAE
from torch.nn import functional as F

def getFeature(dir, years):
    MalwareCorpus=os.path.join(dir, "training", "malware")
    GoodwareCorpus=os.path.join(dir, "training", "goodware")
    AllMalSamples = CM.ListFiles(MalwareCorpus, ".data", year=years)
    AllGoodSamples = CM.ListFiles(GoodwareCorpus, ".data", year=years)
    return AllMalSamples, AllGoodSamples


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
        
def extract_representations(model, features, batch_size=256, device="cpu"):
    model.eval()
    representations = []
    with torch.no_grad():
        for dense_batch in get_batch(features, device=device, batch_size=batch_size):
            encoded_batch, _, _ = model.encode(dense_batch)
            representations.append(encoded_batch)

    return torch.cat(representations, dim=0)


if __name__=="__main__":
    dir = "/home/wang/Data/android"
    device = torch.device("cpu")
    train_years = [str(i) for i in range(2020, 2024)]
    test_years = [str(i) for i in range(2014, 2020)]
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
    svcModel = SVC(kernel='rbf', C=1, gamma='scale')
    svcModel.fit(train_features, train_labels)
    _,_,_,_,_,_ = error.evaluation_metrics(f"Support shift initial", svcModel, test_features, train_features, test_labels, train_labels)
    
    train_labels = torch.tensor(train_labels, dtype=torch.long).to(device)
    
    input_dim = train_features.shape[1]
    assert(input_dim==test_features.shape[1])
    hidden_dim = 256

    model = VanillaVAE(input_dim, hidden_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

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
        running_kld_loss = 0.0

        #test_batch_cycle = cycle(get_batch(test_features, device = device))

        for batch_idx, (dense_batch_train, batch_labels) in enumerate(get_batch(train_features, labels=train_labels, device=device)):
            optimizer.zero_grad()

            #dense_batch_test = next(test_batch_cycle)
            train_encoded, train_decoded, class_logits, mu, log_var = model(dense_batch_train)
            reconstruction_loss_train = F.binary_cross_entropy_with_logits(train_decoded, dense_batch_train, reduction='sum')
            kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            classification_loss_train = classification_loss_function(class_logits, batch_labels)
            total_loss = reconstruction_loss_train + classification_loss_train + kld_loss
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            running_recon_loss_train += reconstruction_loss_train.item()
            running_classification_loss += classification_loss_train.item()
            running_kld_loss += kld_loss.item()

            print(f"Batch {batch_idx + 1}: "
                f"Recon Train Loss: {reconstruction_loss_train.item():.4f}, "
                f"Classification Train Loss: {classification_loss_train.item():.4f}, "
                f"KLD Loss: {kld_loss.item():.4f}, "
                f"Total Loss: {total_loss.item():.4f}")
        
        print(f"Epoch {epoch}: Train Encoded Mean={train_encoded.mean().item()}, Std={train_encoded.std().item()}")
        print(f"Epoch [{epoch+1}/{num_epochs}], "
            f"Total Recon Train Loss: {running_recon_loss_train:.4f}, "
            f"Total KLD Loss: {running_kld_loss:.4f}, "
            f"Total Class Train Loss: {classification_loss_train:.4f}, "
            f"Total Total Loss: {running_loss:.4f}")
        if(running_loss < save_loss):
            save_loss = running_loss
            # torch.save(model.state_dict(), "/home/wang/Data/android/autoencoder_1_rbf_epoch-30—best_hidden-512_batch-256.pth")

    # bmodel = Autoencoder(input_dim, hidden_dim).to(device)
    # bmodel.load_state_dict(torch.load("/home/wang/Data/android/autoencoder_1_rbf_epoch-30—best_hidden-512_batch-256.pth")) 
    # bmodel.to(device)
    del train_features, test_features
    gc.collect()
    
    train_goodware = extract_representations(model, goodfeatures,training=True).detach().cpu().numpy()
    train_malware = extract_representations(model, malfeatures, training=True).detach().cpu().numpy()
    test_goodware = extract_representations(model, tgoodfeatures,training=False).detach().cpu().numpy()
    test_malware = extract_representations(model, tmalfeatures, training=False).detach().cpu().numpy()
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


    test_f1, train_f1, acc, train_acc, test_loss, train_loss = error.evaluation_metrics(f"Support shift after dual encoder", svcModel, test_features, train_features, test_labels, train_labels)
    