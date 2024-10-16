import numpy as np

class PCA :
    def __init__(self, n_components= 2) :
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.std = None
        self.X_standardized = None
        self.corr_coef = None
        self.eigenvalues = None
        self.eigenvectors = None
    
    def fit(self, X) :
        # mean
        self.mean = np.mean(X, axis= 0)
        self.std = np.std(X, axis= 0)
        self.X_standardized = (X - self.mean) / self.std
        # correlation matrix
        self.corr_coef = np.corrcoef(self.X_standardized.T)
        # eigenvalues, eigenvetors
        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.corr_coef)
        # sort eigenvectors
        sorted_indices = np.argsort(self.eigenvalues)[::-1]
        self.eigenvalues = self.eigenvalues[sorted_indices]
        self.eigenvectors = self.eigenvectors[:, sorted_indices]
        # store first n_components eigenvectors
        self.components = self.eigenvectors[:, :self.n_components]

    def transform(self, X) :
        mean = np.mean(X, axis= 0)
        std = np.std(X, axis= 0)
        X = (X - mean) / std
        return np.dot(X, self.components)
    