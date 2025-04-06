'''
Modified from https://github.com/AntixK/PyTorch-VAE.git
'''

import torch
from torch import nn
from torch.nn import functional as F

class VanillaVAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims=None):
        super(VanillaVAE, self).__init__()
        
        self.latent_dim = latent_dim
        if hidden_dims is None:
            hidden_dims = [1024, 512, 256]

        # Encoder
        encoder_layers = []
        last_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(last_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            last_dim = h_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        hidden_dims.reverse()
        decoder_layers = []
        last_dim = latent_dim
        for h_dim in hidden_dims:
            decoder_layers.append(nn.Linear(last_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            last_dim = h_dim
        
        self.decoder = nn.Sequential(*decoder_layers)
        self.final_layer = nn.Linear(hidden_dims[-1], input_dim)

        # Classifier
        self.classifier = nn.Linear(latent_dim, 2)  
        
    def encode(self, x):
        x = self.encoder(x)
        mu, log_var = self.fc_mu(x), self.fc_var(x)
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        x = self.decoder(z)
        return self.final_layer(x)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        class_logits = self.classifier(z)
        return z, self.decode(z), class_logits, mu, log_var
    
    '''
    def loss_function(self, recon_x, x, mu, log_var):
        recon_loss = F.binary_cross_entropy_with_logits(recon_x, x, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + kld_loss
    '''
