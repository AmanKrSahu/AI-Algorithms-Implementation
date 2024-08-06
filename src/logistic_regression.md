# Logistic Regression Algorithm

Logistic Regression is a statistical method for analyzing datasets with one or more independent variables that determine an outcome. It is used for binary classification problems.

## Table of Contents
- [Introduction](#introduction)
- [Algorithm Steps](#algorithm-steps)
- [Mathematical Equations](#mathematical-equations)
- [Gradient Descent](#gradient-descent)
- [Code Implementation](#code-implementation)

## Introduction

Logistic Regression predicts the probability of the target variable belonging to a certain class. Unlike Linear Regression, it models the data using a sigmoid function to output probabilities between 0 and 1.

## Algorithm Steps

1. **Collect the data**: Gather the dataset containing the predictor variables and the target variable.
2. **Select the model**: Choose the logistic model for binary classification.
3. **Fit the model**: Find the coefficients (weights) that maximize the likelihood of the observed data.
4. **Predict**: Use the model to make predictions on new data.
5. **Evaluate**: Assess the model’s performance using appropriate metrics (e.g., accuracy, precision, recall).

## Mathematical Equations

1. Sigmoid Function

    The logistic function (sigmoid function) is defined as:
    $\sigma(z) = \frac{1}{1 + e^{-z}}$

    Where:
    - $z = \beta_0 + \beta_1X_1 + \beta_2X_2 + \ldots + \beta_nX_n $

2. Cost Function (Log Loss)

    The cost function for logistic regression is the log loss (binary cross-entropy):
    $J(\beta) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$

    Where:
    - $( y_i )$ is the actual label.
    - $( \hat{y}_i )$ is the predicted probability.
    - $( n )$ is the number of data points.

## Gradient Descent

Gradient Descent is used to minimize the cost function by iteratively updating the model parameters.

1. Gradient Descent Steps

    1. **Initialize the parameters**: Start with initial values for the coefficients (e.g., zeros or random values).
    2. **Compute the cost function**: Calculate the cost function for the current parameter values.
    3. **Compute the gradient**: Calculate the gradient of the cost function with respect to each parameter.
    4. **Update the parameters**: Adjust the parameters in the opposite direction of the gradient.
    5. **Repeat**: Repeat steps 2-4 until the cost function converges.

2. Gradient Descent Equations

    For each parameter $(\beta_j)$:
    $\beta_j := \beta_j - \alpha \frac{\partial}{\partial \beta_j} J(\beta)$

    Where:
    - $(\alpha)$ is the learning rate.
    - $(\frac{\partial}{\partial \beta_j} J(\beta))$ is the partial derivative of the cost function with respect to $(\beta_j)$.

    The partial derivative of the log loss with respect to $(\beta_j)$ is given by:
    $\frac{\partial}{\partial \beta_j} J(\beta) = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i) X_{ij}$

## Code Implementation

Here is an example of implementing Logistic Regression using Python with the help of the `scikit-learn` library:

```python
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

class LogisticRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(self.n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_pred)

            dw = (1/n_samples) * np.dot(X.T, (predictions-y))
            db = (1/n_samples) * np.sum(predictions-y)

            self.weights = self.weights - self.lr*dw
            self.bias = self.bias - self.lr*db

    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        class_pred = [0 if y<=0.5 else 1 for y in y_pred]
        return class_pred