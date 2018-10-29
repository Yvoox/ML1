import numpy as np
import matplotlib.pyplot as plt
import sys
from template.helpers import *

def sigmoid(t):
    return 1.0 / (1 + np.exp(-t))

def calculate_loss(y, tx, w):
    pred = sigmoid(np.matmul(tx,w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(- loss)

def calculate_gradient(y, tx, w):
    pred = sigmoid(np.matmul(tx,w))
    grad = tx.T.dot(pred - y) / len(pred - y)
    return grad

def learning_by_gradient_descent(y, tx, w, gamma):
    loss = calculate_loss(y, tx, w)
    grad = calculate_gradient(y, tx, w)
    w = w - gamma * grad
    return loss, w

def logistic_regression_GD(y, tx, w, gamma, max_iter):
    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
    return w, loss

# => 79.1652 % with polynomial degree 2
#gamma = 0.01
#max_iters = 7000
