# Random Forest Algorithm

Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.

## Table of Contents
- [Introduction](#introduction)
- [Algorithm Steps](#algorithm-steps)
- [Mathematical Equations](#mathematical-equations)
- [Code Implementation](#code-implementation)

## Introduction

Random Forest combines the predictions of multiple decision trees to produce more accurate and stable predictions. It reduces overfitting by averaging multiple decision trees, trained on different parts of the same training set.

## Algorithm Steps

1. **Select random samples**: Randomly select n samples from the dataset with replacement (bootstrap sampling).
2. **Build decision trees**: For each sample, construct a decision tree using a random subset of features.
3. **Voting/Averaging**: For classification, use the majority voting from all trees. For regression, use the average of all tree outputs.
4. **Output the result**: Combine the results from each decision tree to make the final prediction.

## Mathematical Equations

1. Gini Impurity (for individual trees)

    Gini impurity measures the likelihood of an incorrect classification of a new instance if it was randomly classified according to the distribution of the class labels in the dataset.

    $Gini(D) = 1 - \sum_{i=1}^{C} p_i^2$

    Where:
    - $( p_i )$ is the probability of selecting an element of class $( i )$ in dataset $( D )$.
    - $( C )$ is the number of classes.

2. Entropy (for individual trees)

    Entropy measures the impurity or uncertainty in the dataset.

    $Entropy(D) = - \sum_{i=1}^{C} p_i \log_2(p_i)$

    Where:
    - $( p_i )$ is the probability of selecting an element of class $( i )$ in dataset $( D )$.
    - $( C )$ is the number of classes.

3. Information Gain (for individual trees)

    Information Gain is the reduction in entropy after a dataset is split on an attribute.

    $IG(D, A) = Entropy(D) - \sum_{v \in Values(A)} \frac{|D_v|}{|D|} Entropy(D_v)$

    Where:
    - $( D )$ is the dataset.
    - $( A )$ is the attribute.
    - $( D_v )$ is the subset of $( D )$ for which attribute $( A )$ has value $( v )$.

## Code Implementation

Here is an example of implementing a Random Forest using Python with the help of the `scikit-learn` library:

```python
import numpy as np
from collections import Counter
from DecisionTree import DecisionTree

class RandomForest:
    def __int__(self, n_trees=10, max_depth=10, min_samples_split=2, n_feature=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_feature
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(
                max_depth = self.max_depth,
                min_samples_split = self.min_samples_split,
                n_features = self.n_features
            )
            X_sample, y_sample = self._bootstrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_samples(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self._most_common_label(tree_pred) for pred in tree_preds])
        return predictions

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common