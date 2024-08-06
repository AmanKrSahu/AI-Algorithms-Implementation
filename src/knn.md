# K-Nearest Neighbors (KNN) Algorithm

K-Nearest Neighbors (KNN) is a simple, supervised machine learning algorithm that can be used for both classification and regression tasks. It operates on the principle that similar instances are likely to be near each other.

## Table of Contents
- [Introduction](#introduction)
- [Algorithm Steps](#algorithm-steps)
- [Mathematical Equations](#mathematical-equations)
- [Code Implementation](#code-implementation)
- [Conclusion](#conclusion)

## Introduction

KNN is a type of instance-based learning, or lazy learning, where the function is only approximated locally and all computations are deferred until function evaluation. 

## Algorithm Steps

1. **Choose the number of K and a distance metric**: The number of neighbors \( K \) and the distance metric (e.g., Euclidean, Manhattan) are chosen.
2. **Calculate the distance between the query-instance and all the training samples**: Using the chosen distance metric.
3. **Sort the distances**: Sort the distances from smallest to largest.
4. **Select the K nearest neighbors**: Pick the first K entries from the sorted list.
5. **Determine the most frequent class (for classification) or average (for regression)**: Based on the labels of the K nearest neighbors.

## Mathematical Equations

1. Distance Metrics

    - **Euclidean Distance**: 
    $d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$
    
    - **Manhattan Distance**: 
    $d(p, q) = \sum_{i=1}^{n} |p_i - q_i|$

2.  Classification Decision Rule

    **Majority Vote**: For classification, the class label with the highest frequency among the K nearest neighbors is chosen.

    $\hat{y} = \arg\max_{y \in Y} \sum_{i=1}^{K} \mathbb{1}(y_i = y)$

3. Regression Decision Rule

    **Average**: For regression, the average of the K nearest neighbors' values is taken.

    $\hat{y} = \frac{1}{K} \sum_{i=1}^{K} y_i$

## Code Implementation

Here is an example of implementing the KNN algorithm using Python with the help of the `scikit-learn` library:

```python
import numpy as np
from collections import Counter

def euclidian_distance(x1, x2):
    distance = np.sqrt(np.sum((x1-x2)**2))
    return distance

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        distances = [euclidian_distance(x, x_train) for x_train in self.X_train]

        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]