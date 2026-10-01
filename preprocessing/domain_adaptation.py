import numpy as np
from scipy.linalg import inv, sqrtm


def euclidean_alignment(X):
    n_trials, n_channels, _ = X.shape
    covs = np.zeros((n_trials, n_channels, n_channels))

    for i in range(n_trials):
        covs[i] = np.cov(X[i])

    r_bar = np.mean(covs, axis=0)
    r_inv_sqrt = np.real(inv(sqrtm(r_bar)))

    X_aligned = np.zeros_like(X)
    for i in range(n_trials):
        X_aligned[i] = np.matmul(r_inv_sqrt, X[i])

    return X_aligned
