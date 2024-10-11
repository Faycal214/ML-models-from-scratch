import numpy as np
from collections import Counter

# Define a class for the node of the decision tree
class Node:
    # Initialize a node with information about the feature, threshold, left/right children, and value (if it's a leaf node)
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature  # Feature index for splitting
        #the threshold helps the decision tree to decide how to split the data based on the selected feature at each node
        # making the tree more discriminative in classifying the data
        self.threshold = threshold 
        self.left = left  # Left child node
        self.right = right  # Right child node
        self.value = value  # Value for a leaf node (classification label)
    
    # Check if the current node is a leaf (has a value instead of children)
    def is_leaf_node(self):
        return self.value is not None


# Define the Decision Tree class
class DecisionTree:
    # Initialize the decision tree with parameters like minimum samples to split, maximum depth, and number of features
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        # Minimum number of samples required to split a node. If fewer samples remain, it will stop splitting
        self.min_samples_split = min_samples_split
        # The maximum depth allowed for the tree
        self.max_depth = max_depth
        # The number of features to consider when splitting the data (can use all or a subset)
        self.n_features = n_features
        # Root of the tree, initially set to None
        self.root = None
    
    # Trains the decision tree on the data (X features, y labels) by recursively building (growing) the tree
    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)  # Determine number of features
        self.root = self._grow_tree(X, y)  # Grow the tree recursively starting from the root

    # Builds the tree recursively, it checks for stopping criteria like maximum depth or if the labels are pure (i.e., all the same)
    # then either creates a leaf node or splits further
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape  # Number of samples and features in the dataset
        n_labels = len(np.unique(y))  # Number of unique labels (classes)

        # Check if we should stop (max depth reached, only one class left, or not enough samples to split)
        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)  # Assign the most common label as the leaf node value
            return Node(value=leaf_value)  # Return a leaf node
        
        # Randomly select features to consider for splitting
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        
        # Find the best feature and threshold to split the dataset
        best_feature, best_threshold = self._best_split(X, y, feat_idxs)

        # Split the dataset into left and right child nodes
        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)  # Recursively grow the left child
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)  # Recursively grow the right child
        
        return Node(best_feature, best_threshold, left, right)  # Return a node with the feature, threshold, and children
    
    # If the tree stops splitting, it creates a leaf node with the most common label in the subset of data
    def _most_common_label(self, y):
        counter = Counter(y)  # Count occurrences of each label
        value = counter.most_common(1)[0][0]  # Get the most common label
        return value
    
    # Splits the data into two parts: one for values <= threshold and another for values > threshold
    def _split(self, X_column, split_threshold):
        left_idxs = np.argwhere(X_column <= split_threshold).flatten()  # Indices where values are less than or equal to the threshold
        right_idxs = np.argwhere(X_column > split_threshold).flatten()  # Indices where values are greater than the threshold
        return left_idxs, right_idxs  # Return the indices for left and right splits
    
    # Finds the best feature and threshold to split the data by maximizing the information gain
    # It goes through each feature and tries various thresholds to find the split that results in the highest information gain
    def _best_split(self, X, y, feat_idxs):
        best_gain = -1  # Initialize the best information gain
        split_idx, split_threshold = None, None  # Store the best feature index and threshold

        # Iterate over the selected features
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]  # Get the values of the current feature
            thresholds = np.unique(X_column)  # Get unique values (potential thresholds)

            # Iterate over thresholds and compute the information gain
            for thr in thresholds:
                gain = self._information_gain(y, X_column, thr)  # Calculate information gain for the split

                if gain > best_gain:  # Update if the current gain is better than the best one
                    best_gain = gain
                    split_idx = feat_idx  # Store the index of the best feature
                    split_threshold = thr  # Store the best threshold
        
        return split_idx, split_threshold  # Return the best feature and threshold
    
    # Calculates the information gain for a specific feature and threshold
    # It compares the entropy before and after the split to see how much "disorder" was reduced
    def _information_gain(self, y, X_column, threshold):
        parent_entropy = self._entropy(y)  # Entropy of the parent node
        left_idxs, right_idxs = self._split(X_column, threshold)  # Split the data into left and right based on the threshold

        # If one of the splits is empty, return no gain
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        
        # Calculate the weighted entropy of the children nodes
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r  # Weighted average of child entropies

        # Calculate the information gain (parent entropy - child entropy)
        information_gain = parent_entropy - child_entropy
        return information_gain
    
    # Calculate the entropy of a label distribution (how impure a node is)
    def _entropy(self, y):
        hist = np.bincount(y)  # Count occurrences of each label
        ps = hist / len(y)  # Calculate probabilities
        return -np.sum([p * np.log(p) for p in ps if p > 0])  # Calculate entropy using log(p)
    
    
    # Predict labels for the given dataset
    def predict(self, X):
        # Traverse the tree for each sample in X and make a prediction
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    # a helper function to navigate through the tree
    # for each input x, it checks the feature and threshold of the current node and moves left or right depending on the value of x
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():  # If the node is a leaf, return its value (label)
            return node.value
        
        # Traverse left or right based on the feature value and threshold
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)  # Go left
        return self._traverse_tree(x, node.right)  # Go right
