# Naive Bayes Algorithm

Naive Bayes is a family of simple yet effective probabilistic classifiers based on Bayes' Theorem with the "naive" assumption of conditional independence between every pair of features given the class label.

## Table of Contents
- [Introduction](#introduction)
- [Algorithm Steps](#algorithm-steps)
- [Mathematical Equations](#mathematical-equations)
- [Code Implementation](#code-implementation)

## Introduction

Naive Bayes classifiers are highly scalable and efficient for both binary and multi-class classification tasks. They are particularly well-suited for text classification problems such as spam detection, sentiment analysis, and document categorization.

## Algorithm Steps

1. **Calculate prior probabilities**: Determine the prior probability for each class in the training dataset.
2. **Calculate likelihoods**: For each class, calculate the likelihood of each feature given the class.
3. **Apply Bayes' Theorem**: Use Bayes' Theorem to calculate the posterior probability for each class given the features of a new data point.
4. **Predict the class**: Assign the class with the highest posterior probability to the new data point.

## Mathematical Equations

1. Bayes' Theorem

    $P(C_k|X) = \frac{P(X|C_k) P(C_k)}{P(X)}$

    Where:
    - $( P(C_k|X) )$ is the posterior probability of class $( C_k )$ given feature vector $( X )$.
    - $( P(X|C_k) )$ is the likelihood of feature vector ( X ) given class $( C_k )$.
    - $( P(C_k) )$ is the prior probability of class $( C_k )$.
    - $( P(X) )$ is the evidence, which is the probability of the feature vector $( X )$ across all classes.

2. Naive Bayes Assumption

    The naive assumption is that features are conditionally independent given the class label:

    $P(X|C_k) = \prod_{i=1}^{n} P(x_i|C_k)$

    Where:
    - $( x_i )$ is the $( i )$-th feature.
    - $( n )$ is the number of features.

3. Posterior Probability

    Combining Bayes' Theorem with the naive assumption:

    $P(C_k|X) \propto P(C_k) \prod_{i=1}^{n} P(x_i|C_k)$

## Code Implementation

Here is an example of implementing Naive Bayes using Python with the help of the `scikit-learn` library:

```python
import numpy as np

class NaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # Calculate the mean, variance & prior for each class
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0]/float(n_samples)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []

        # Calculate posterior probabilities for each class
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posteriors = np.sum(np.log(self._pdf(idx, x)))
            posterior = posterior + prior
            posteriors.append(posterior)

        # Return class with the highest posterior
        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((x-mean)**2) / (2*var))
        denominator = np.sqrt(2*np.pi*var)
        return numerator/denominator