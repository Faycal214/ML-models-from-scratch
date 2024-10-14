import numpy as np
from decisionTree import *
from collections import Counter


class RandomForest :
    def __init__(self, n_trees = 10, max_depth = 10, min_sample_split = 2, n_features = None) :
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y) :
        self.trees = []
        for _ in range(self.n_trees) :
            tree = DecisionTree(max_depth= self.max_depth,
                         min_samples_split= self.min_sample_split, 
                         n_features= self.n_features)
            
            X_sample, y_sample = self._get_subset_from_dataset(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
    

    def _get_subset_from_dataset(self, X, y) :
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace= True)
        return X[idxs], y[idxs]
    
    # If the tree stops splitting, it creates a leaf node with the most common label in the subset of data
    def _most_common_label(self, y):
        counter = Counter(y)  # Count occurrences of each label
        value = counter.most_common(1)[0][0]  # Get the most common label
        return value
    
    def predict(self, X) :
        predictions = np.array([tree.predict(X) for tree in self.trees])
        trees_pred = np.swapaxes(predictions, 0 ,1)
        return np.array([self._most_common_label(pred) for pred in trees_pred])
