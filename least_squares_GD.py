import numpy as np
import matplotlib.pyplot as plt
import sys
from template.helpers import *

def calculate_loss(y, tx, w):
    e = y - np.dot(tx, w)
    loss= (1/2)*np.mean(e*e)
    return loss

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    err = y - tx.dot(w)
    grad = - tx.T.dot(err) / len(err)
    return grad

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """
    grad = calculate_gradient(y, tx, w)
    w = w - gamma*grad
    return w

def regression_GD(y, tx, w, gamma, max_iter):
    for iter in range(max_iter):
        # get loss and update w.
        w = learning_by_gradient_descent(y, tx, w, gamma)
        loss = calculate_loss(y, tx, w)
    return w, loss

# => 75.796399%  with polynomial degree 2
#gamma = 0.001
#max_iters = 5000
