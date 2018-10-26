import numpy as np
import matplotlib.pyplot as plt
import sys
from template.helpers import *

def calculate_loss(y, tx, w):
    e = y - tx.dot(w)
    loss = 1/2*np.mean(e**2)
    return loss

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_loss(y, tx, w)
    grad = calculate_gradient(y, tx, w)
    w -= gamma * grad
    return loss, w

def gradient_descent(y, tx, w, gamma, max_iter, threshold):
    losses = []

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w
