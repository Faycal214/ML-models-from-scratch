import numpy as np
from collections import Counter
from scipy.spatial import KDTree  # Use KDTree for faster neighbor search
from joblib import Parallel, delayed  # For parallel processing
from sklearn.model_selection import GridSearchCV

# =========K-Nearest neighbors algorithm==========

# n_neighbors :int, default=5
# Number of neighbors to use by default for kneighbors queries

# weights{‘uniform’, ‘distance’}, callable or None, default=’uniform’
# Weight function used in prediction. Possible values:
# ‘uniform’ : uniform weights. All points in each neighborhood are weighted equally
# ‘distance’ : weight points by the inverse of their distance. in this case, closer neighbors of a query point will have a greater influence than neighbors which are further away
# [callable] : a user-defined function which accepts an array of distances, and returns an array of the same shape containing the weights

# algorithm{‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, default=’auto’
# Algorithm used to compute the nearest neighbors:
# ‘ball_tree’ will use BallTree
# ‘kd_tree’ will use KDTree
# ‘brute’ will use a brute-force search
# ‘auto’ will attempt to decide the most appropriate algorithm based on the values passed to fit method

# leaf_size : int, default=30
# Leaf size passed to BallTree or KDTree
# This can affect the speed of the construction and query, as well as the memory required to store the tree
# The optimal value depends on the nature of the problem

# p : float, default=2
# Power parameter for the Minkowski metric
# When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2
# For arbitrary p, minkowski_distance (l_p) is used. This parameter is expected to be positive

# metric : str or callable, default=’minkowski’
# Metric to use for distance computation, default is “minkowski”, which results in the standard Euclidean distance when p = 2

# Parallelization using Joblib:
# the predict method is parallelized using the joblib library, by default, the number of jobs (n_jobs) is set to 1 (no parallelization)
# but you can set it to -1 to use all available cores for maximum parallelism or any other number for specific cores
# for each test sample, the distance to all training points is computed in parallel, making it faster for large datasets

class KNN :
    def __init__(self, n_neighbors= 5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', n_jobs= 1) :
        self.n_neighbors= n_neighbors
        self.weights= weights
        self.algorithm= algorithm
        self.leaf_size= leaf_size
        self.p= p
        self.metric= metric
        self.n_jobs = n_jobs  # Number of parallel jobs
        self.tree = None
        self.dist_func = self._get_distance_func()
    
    # Different distance functions
    def _get_distance_func(self):
        if self.metric == 'minkowski':
            return lambda x1, x2: np.sum(np.abs(x1 - x2) ** self.p) ** (1 / self.p)
        elif self.metric == 'euclidean':
            return lambda x1, x2: np.sqrt(np.sum(x2 - x1)**2)
        elif self.metric == 'manhattan':
            return lambda x1, x2: np.sum(np.abs(x1 - x2))
        elif callable(self.metric):  # User-defined distance function
            return self.metric
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
    

    # Compute the weights for each neighbor if applicable
    def _get_weights(self, distances):
        if self.weights == 'uniform':
            return np.ones_like(distances)
        elif self.weights == 'distance':
            return 1 / (distances + 1e-5)  # Avoid division by zero
        elif callable(self.weights):
            return self.weights(distances)
        else:
            raise ValueError(f"Unsupported weight type: {self.weights}")

    
    def _predict(self, x):
        # Use KDTree or brute-force search for neighbors
        if self.tree is not None:
            distances, indices = self.tree.query(x, k=self.n_neighbors)
        else:
            distances = [self.dist_func(x, x_train) for x_train in self.X_train]
            indices = np.argsort(distances)[:self.n_neighbors]
            distances = np.array(distances)[indices]  # Reorder distances

        # Get the labels of the nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in indices]
        
        # Apply weights if necessary
        weights = self._get_weights(distances)
        
        # Perform weighted voting
        weighted_votes = Counter()
        for label, weight in zip(k_nearest_labels, weights):
            weighted_votes[label] += weight

        # Return the label with the highest vote (most common)
        return weighted_votes.most_common(1)[0][0]
    
    # Fit the k-nearest neighbors classifier from the training dataset
    def fit(self, X, y) :
        self.X_train= X
        self.y_train= y
        if self.algorithm == 'kd_tree':
            self.tree = KDTree(X, leafsize=self.leaf_size)

    
    # Predict the class labels for the provided data
    def predict(self, X) :
        predictions = Parallel(n_jobs=self.n_jobs)(delayed(self._predict)(x) for x in X)
        return np.array(predictions)

    # Mean accuracy
    def score(self, y_true, y_pred):
        return np.sum(y_pred == y_true) / len(y_true)
    
