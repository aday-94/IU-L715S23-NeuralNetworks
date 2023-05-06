""" NN-L715 Spring 2023 Wk5 Practical Andrew Davis - Logistic Regression (https://cl.indiana.edu/~ftyers/courses/2023/Spring/L-715/practicals/gradient/gradient.html) """

import numpy as np

# setting the parameters

W = np.random.rand(3)
B = np.random.rand(1)
print(W, B)

learning_rate = 0.1
num_epochs = 50

def sigmoid(z):
    sigma = 1/(1+np.exp(-z))
    return np.round(sigma, decimals=4)

def predict(X, W, B):
    z = np.matmul(X, np.transpose(W)+B)
    y_hat = sigmoid(z)
    return y_hat

def cost_function(y_hat, Y):
    m = len(Y)
    y_log_y_hat = np.multiply(np.log(y_hat), Y)
    loss = -np.sum(y_log_y_hat)
    return loss

def partial_derivative_w(X, diff):
    return np.matmul(np.transpose(diff), X)

def partial_derivative_b(diff):
    return np.sum(diff, axis=0)

def update_hyperparameter: