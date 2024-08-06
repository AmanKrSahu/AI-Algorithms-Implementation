# Decision Tree Algorithm

Decision Tree is a supervised learning algorithm used for both classification and regression tasks. It splits the data into subsets based on the most significant attribute that makes the best decision at each node.

## Table of Contents
- [Introduction](#introduction)
- [Algorithm Steps](#algorithm-steps)
- [Mathematical Equations](#mathematical-equations)
- [Code Implementation](#code-implementation)

## Introduction

A Decision Tree is a tree-like model where each internal node represents a test on an attribute, each branch represents an outcome of the test, and each leaf node represents a class label or a continuous value.

## Algorithm Steps

1. **Select the best attribute**: Choose the attribute that best separates the data based on a chosen criterion (e.g., Gini impurity, entropy).
2. **Create a decision node**: Split the dataset into subsets based on the chosen attribute.
3. **Repeat**: Recursively apply the above steps to each subset until one of the stopping conditions is met (e.g., all data points belong to the same class, maximum depth is reached).
4. **Assign a class label**: Assign the most frequent class in the subset to the leaf node (for classification) or the average value (for regression).

## Mathematical Equations

1. Gini Impurity

    Gini impurity measures the likelihood of an incorrect classification of a new instance if it was randomly classified according to the distribution of the class labels in the dataset.

    $Gini(D) = 1 - \sum_{i=1}^{C} p_i^2$

    Where:
    - $( p_i )$ is the probability of selecting an element of class $( i )$ in dataset $( D )$.
    - $( C )$ is the number of classes.

2. Entropy

    Entropy measures the impurity or uncertainty in the dataset.

    $Entropy(D) = - \sum_{i=1}^{C} p_i \log_2(p_i)$

    Where:
    - $( p_i )$ is the probability of selecting an element of class $( i )$ in dataset $( D )$.
    - $( C )$ is the number of classes.

3. Information Gain

    Information Gain is the reduction in entropy after a dataset is split on an attribute.

    $IG(D, A) = Entropy(D) - \sum_{v \in Values(A)} \frac{|D_v|}{|D|} Entropy(D_v)$

    Where:
    - $( D )$ is the dataset.
    - $( A )$ is the attribute.
    - $( D_v )$ is the subset of $( D )$ for which attribute $( A )$ has value $( v )$.

## Code Implementation

Here is an example of implementing a Decision Tree using Python with the help of the `scikit-learn` library:

```python
import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # Check the Stopping Criteria
        if(depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        # Find the Best Split
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        # Create Child Nodes
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_feature, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx, in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique[X_column]

            for thr in thresholds:
                # Calculate the Information Gain
                gain = self._information_gain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold

    def _information_gain(self, y, X_column, threshold):
        # Parent Entropy
        parent_entropy = self._entropy(y)

        # Create Children
        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0 

        # Calculate the weighted average entropy of children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self.entropy(y[left_idxs]), self.entropy(y[right_idxs])
        child_entropy = (n_l/n)*e_l + (n_r/n)*e_r 

        # Calculate the IG
        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p*np.log(p) for p in ps if p>0])

    def _most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        
        return self._traverse_tree(x, node.right)