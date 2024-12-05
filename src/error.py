import os
import numpy as np


def sample_spherical_gaussian_from_w(w, sigma, num_samples):
    cov_matrix = np.power(sigma, 2) * np.eye(len(w))
    w_samples = np.random.multivariate_normal(w, cov_matrix, size=num_samples)
    return w_samples

def zero_one_loss(y_true, y_pred):
    return np.mean(y_true != y_pred)

def get_loss(model, w_samples, x, y_true):
    losses = [zero_one_loss(y_true, model.predict(x, w_prime)) for w_prime in w_samples]
    avg_loss = np.mean(losses)
    std_loss = np.std(losses)
    return avg_loss, std_loss
