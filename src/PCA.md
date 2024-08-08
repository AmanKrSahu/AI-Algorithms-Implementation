# Principal Component Analysis (PCA)

Principal Component Analysis (PCA) is an unsupervised dimensionality reduction technique used to reduce the number of features in a dataset while preserving as much variability as possible.

## Table of Contents
- [Introduction](#introduction)
- [Algorithm Steps](#algorithm-steps)
- [Mathematical Equations](#mathematical-equations)
- [Code Implementation](#code-implementation)

## Introduction

PCA transforms the original variables into a new set of uncorrelated variables called principal components, ordered by the amount of variance they capture from the data. It is commonly used for data compression, noise reduction, and visualization.

## Algorithm Steps

1. **Standardize the data**: Ensure each feature has a mean of 0 and a standard deviation of 1.
2. **Compute the covariance matrix**: Measure the covariances between every pair of features in the dataset.
3. **Calculate the eigenvalues and eigenvectors**: Derive the principal components by solving the eigenvalue problem of the covariance matrix.
4. **Sort the eigenvalues and eigenvectors**: Rank the eigenvalues in descending order and sort the eigenvectors accordingly.
5. **Select the top k eigenvectors**: Choose the top k eigenvectors corresponding to the k largest eigenvalues to form a new matrix.
6. **Transform the data**: Project the original data onto the new k-dimensional subspace.

## Mathematical Equations

1. Standardization

    Standardize each feature $( x_i )$ to have a mean of 0 and a standard deviation of 1:

    $z_i = \frac{x_i - \mu_i}{\sigma_i}$

    Where:
    - $( x_i )$ is the original feature.
    - $( \mu_i )$ is the mean of the feature.
    - $( \sigma_i )$ is the standard deviation of the feature.
    - $( z_i )$ is the standardized feature.

2. Covariance Matrix

    Compute the covariance matrix $( \mathbf{C} )$ for the standardized data:

    $\mathbf{C} = \frac{1}{n-1} \sum_{i=1}^{n} (\mathbf{z}_i - \mathbf{\mu})(\mathbf{z}_i - \mathbf{\mu})^T$

    Where:
    - $( \mathbf{z}_i )$ is the standardized data point.
    - $( \mathbf{\mu} )$ is the mean of the standardized data.
    - $( n )$ is the number of data points.

3. Eigenvalues and Eigenvectors

    Solve the eigenvalue problem for the covariance matrix $( \mathbf{C} )$:

    $\mathbf{C} \mathbf{v}_i = \lambda_i \mathbf{v}_i$

    Where:
    - $( \mathbf{v}_i )$ is the eigenvector.
    - $( \lambda_i )$ is the eigenvalue.

4. Projection

    Project the original data onto the new subspace defined by the top k eigenvectors:

    $\mathbf{Z} = \mathbf{X} \mathbf{W}$

    Where:
    - $( \mathbf{Z} )$ is the projected data.
    - $( \mathbf{X} )$ is the original data.
    - $( \mathbf{W} )$ is the matrix of the top k eigenvectors.

## Code Implementation

Here is an example of implementing PCA using Python with the help of the `scikit-learn` library:

```python
import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # Mean Centering
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # Covariance; functions needs samples as columns
        cov = np.cov(X.T)

        # EigenVectors, EigenValues
        eigenvectors, eigenvalues = np.linalg.eig(cov)

        # EigenVectors v = [:, i] column vector; transpose this for easier calculations
        eigenvectors = eigenvectors.T

        # Sort EigenVectors
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        self.components = eigenvectors[:self.n_components]

    def transform(self, X):
        # Projects Data
        X = X - self.mean
        return np.dot(X, self.components.T)