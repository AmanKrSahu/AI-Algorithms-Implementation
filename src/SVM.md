# Support Vector Machine (SVM)

Support Vector Machine (SVM) is a powerful supervised learning algorithm used for both classification and regression tasks. SVMs are particularly effective in high-dimensional spaces and are widely used in text classification, image recognition, and bioinformatics.

## Table of Contents
- [Introduction](#introduction)
- [Algorithm Steps](#algorithm-steps)
- [Mathematical Equations](#mathematical-equations)
- [Code Implementation](#code-implementation)

## Introduction

SVM works by finding the hyperplane that best separates the classes in the feature space. The optimal hyperplane is the one that maximizes the margin between the nearest data points of both classes, known as support vectors.

## Algorithm Steps

1. **Select a kernel function**: Choose a kernel function to transform the input data into a higher-dimensional space if necessary.
2. **Construct the optimal hyperplane**: Calculate the hyperplane that maximizes the margin between the two classes.
3. **Classify data**: Use the hyperplane to classify new data points.

## Mathematical Equations

1. Hyperplane Equation

    The equation of a hyperplane in a d-dimensional space is given by:

    $\mathbf{w} \cdot \mathbf{x} + b = 0$

    Where:
    - $( \mathbf{w} )$ is the weight vector.
    - $( \mathbf{x} )$ is the feature vector.
    - $( b )$ is the bias term.

2. Decision Function

    The decision function that determines on which side of the hyperplane a data point lies is:

    $f(\mathbf{x}) = \text{sign}(\mathbf{w} \cdot \mathbf{x} + b)$

3. Margin

    The margin is the distance between the hyperplane and the nearest data point from either class. The margin \( M \) is defined as:

    $M = \frac{2}{\|\mathbf{w}\|}$

4. Optimization Objective

    The SVM optimization problem is to minimize the following objective function:

    $\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2$

    Subject to the constraint:

    $y_i (\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1 \quad \text{for all } i$

    Where:
    - $( y_i )$ is the class label of the i-th data point.
    - $( \mathbf{x}_i )$ is the feature vector of the i-th data point.

5. Kernel Trick

    For non-linearly separable data, the kernel trick is used to map the data into a higher-dimensional space:

    $K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i) \cdot \phi(\mathbf{x}_j)$

    Where:
    - $( K )$ is the kernel function.
    - $( \phi(\mathbf{x}) )$ is the transformation function.

    Common kernel functions include:
    - **Linear Kernel**: $( K(\mathbf{x}_i, \mathbf{x}_j) = \mathbf{x}_i \cdot \mathbf{x}_j )$
    - **Polynomial Kernel**: $( K(\mathbf{x}_i, \mathbf{x}_j) = (\mathbf{x}_i \cdot \mathbf{x}_j + c)^d )$
    - **Radial Basis Function (RBF) Kernel**: $( K(\mathbf{x}_i, \mathbf{x}_j) = \exp(-\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2) )$

## Code Implementation

Here is an example of implementing SVM using Python with the help of the `scikit-learn` library:

```python
import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        y_ = np.where(y <= 0, -1, 1)

        # Initialising Weights
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)