# Perceptron Algorithm

The Perceptron is a type of linear classifier, and one of the simplest types of artificial neural networks. It is a supervised learning algorithm used for binary classification tasks.

## Table of Contents
- [Introduction](#introduction)
- [Algorithm Steps](#algorithm-steps)
- [Mathematical Equations](#mathematical-equations)
- [Code Implementation](#code-implementation)

## Introduction

The Perceptron algorithm is a foundational building block in machine learning. It models a neuron in a neural network and can classify data into two distinct classes. The algorithm learns a decision boundary by adjusting weights based on the input data.

## Algorithm Steps

1. **Initialize weights**: Start with small random weights and a bias.
2. **Feed the input**: Compute the weighted sum of the inputs and the bias.
3. **Activation function**: Apply the activation function to the weighted sum (typically a step function).
4. **Update weights**: Adjust the weights based on the error (difference between predicted and actual output).
5. **Repeat**: Continue the process until the model converges or for a fixed number of iterations.

## Mathematical Equations

1. Weighted Sum

    The weighted sum $( z )$ of inputs is calculated as:

    $z = \mathbf{w} \cdot \mathbf{x} + b$

    Where:
    - $( \mathbf{w} )$ is the weight vector.
    - $( \mathbf{x} )$ is the input feature vector.
    - $( b )$ is the bias term.

2. Activation Function

    A common activation function for a Perceptron is the step function:

    $y = \begin{cases} 
    1 & \text{if } z \geq 0 \\
    0 & \text{if } z < 0 
    \end{cases}$

3. Weight Update Rule

    The weights are updated using the following rule:

    $\mathbf{w} \leftarrow \mathbf{w} + \eta \times (y_{\text{true}} - y_{\text{pred}}) \times \mathbf{x}$

    $b \leftarrow b + \eta \times (y_{\text{true}} - y_{\text{pred}})$

    Where:
    - $( \eta )$ is the learning rate.
    - $( y_{\text{true}} )$ is the actual label.
    - $( y_{\text{pred}} )$ is the predicted label.

## Code Implementation

Here is an example of implementing a Perceptron using Python with the help of the `scikit-learn` library:

```python
import numpy as np

def unit_step_func(x):
    return np.where(x > 0 , 1, 0)

class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = unit_step_func
        self.weights = None
        self.bias = None


    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialise Parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.where(y > 0 , 1, 0)

        # Learning Weights
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)

                # Perceptron Update Rule
                update = self.lr * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted