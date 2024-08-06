# Linear Regression Algorithm

Linear Regression is a simple and commonly used algorithm in supervised learning for predicting a continuous target variable based on one or more predictor variables.

## Table of Contents
- [Introduction](#introduction)
- [Algorithm Steps](#algorithm-steps)
- [Mathematical Equations](#mathematical-equations)
- [Gradient Descent](#gradient-descent)
- [Code Implementation](#code-implementation)

## Introduction

Linear Regression establishes a relationship between the dependent variable (Y) and one or more independent variables (X) using a linear equation. It aims to find the best-fitting straight line through the data points.

## Algorithm Steps

1. **Collect the data**: Gather the dataset containing the predictor variables and the target variable.
2. **Select the model**: Choose the linear model, which can be simple (one predictor) or multiple (more than one predictor).
3. **Fit the model**: Find the coefficients (weights) that minimize the cost function (e.g., Mean Squared Error).
4. **Predict**: Use the model to make predictions on new data.
5. **Evaluate**: Assess the model’s performance using appropriate metrics (e.g., R-squared, Mean Squared Error).

## Mathematical Equations

1. Linear Equation

    - For a single predictor variable:
    $Y = \beta_0 + \beta_1X$
    
    - For multiple predictor variables:
    $Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + \ldots + \beta_nX_n$

2. Cost Function (Mean Squared Error)

    $MSE = \frac{1}{n} \sum_{i=1}^{n} (Y_i - \hat{Y_i})^2$

    Where:
    - $( Y_i )$ is the actual value.
    - $( \hat{Y_i} )$ is the predicted value.
    - $( n )$ is the number of data points.

3. Coefficient Estimation (Ordinary Least Squares)

    $\beta = (X^TX)^{-1}X^TY$

    Where:
    - $( X )$ is the matrix of predictor variables.
    - $( Y )$ is the vector of target variables.

## Gradient Descent

Gradient Descent is an optimization algorithm used to minimize the cost function by iteratively adjusting the model parameters. 

1. Gradient Descent Steps

    1. **Initialize the parameters**: Start with initial values for the coefficients (e.g., zeros or random values).
    2. **Compute the cost function**: Calculate the cost function for the current parameter values.
    3. **Compute the gradient**: Calculate the gradient of the cost function with respect to each parameter.
    4. **Update the parameters**: Adjust the parameters in the opposite direction of the gradient.
    5. **Repeat**: Repeat steps 2-4 until the cost function converges.

2. Gradient Descent Equations

    For each parameter $(\beta_j)$:
    $\beta_j := \beta_j - \alpha \frac{\partial}{\partial \beta_j} MSE$

    Where:
    - $(\alpha)$ is the learning rate.
    - $(\frac{\partial}{\partial \beta_j} MSE)$ is the partial derivative of the cost function with respect to $(\beta_j)$.

    The partial derivative of the MSE with respect to $(\beta_j)$ is given by:
    $\frac{\partial}{\partial \beta_j} MSE = -\frac{2}{n} \sum_{i=1}^{n} (Y_i - \hat{Y_i}) X_{ij}$

## Code Implementation

Here is an example of implementing Linear Regression using Python with the help of the `scikit-learn` library:

```python
import numpy as np

class LinearRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias

            dw = (1/n_samples) * np.dot(X.T, (y_pred-y))
            db = (1/n_samples) * np.sum(y_pred-y)

            self.weights = self.weights - self.lr*dw
            self.bias = self.bias - self.lr*db

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred