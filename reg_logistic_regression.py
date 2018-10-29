import numpy as np
import matplotlib.pyplot as plt
import sys
from template.helpers import *

def sigmoid(t):
    """apply sigmoid function on t."""
    return 1.0 / (1 + np.exp(-t))

def calculate_loss(y, tx, w):
    pred = sigmoid(np.matmul(tx,w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(- loss)

def calculate_gradient(y, tx, w):
    pred = sigmoid(np.matmul(tx,w))
    grad = tx.T.dot(pred - y) / len(pred - y)
    return grad

def regulized_logistic_regression(y, tx, w, lambda_):
    """return the loss and gradient."""
    loss = calculate_loss(y, tx, w) + lambda_ * (w.T.dot(w)).mean()
    gradient = calculate_gradient(y, tx, w) + 2 * lambda_ * w
    return loss, gradient

def learning_by_regulized_gradient(y, tx, w, gamma, lambda_):

    loss, gradient = regulized_logistic_regression(y, tx, w, lambda_)
    w -= gamma * gradient
    return loss, w

def reg_logistic_regression_GD(y, tx, w, gamma, max_iter, lambda_):
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_regulized_gradient(y, tx, w, gamma, lambda_)
    return w, loss

# => 75.714 % with polynomial degree 2
#gamma = 0.1
#lambda_ = 0.1
#max_iters = 7000
