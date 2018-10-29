import numpy as np
import matplotlib.pyplot as plt
from template.helpers import *



#Least Squares Stochastic Gradient Descent
#-- y : true output values
#-- tx : X samples
#-- initial_w : initial vector w
#-- max_iters : maximum iteration number
#-- gamma : iteration step




def batch_training_set(y,tx,batch_size):
    batch_y = []
    batch_tx = []
    for it in range(batch_size):
        random = np.random.randint(len(y), size=1)
        batch_y = y[random]
        batch_tx = tx[random]
    return batch_y,batch_tx

def grad_eval(y, tx, w):
    e = y - tx.dot(w)
    grad = -tx.T.dot(e) / len(e)
    return grad



def least_squares_SGD(y,tx,initial_w,max_iters,gamma):
    batch_size =1
    w = initial_w
    loss = 10000

    for x in range(max_iters):
        batch_y,batch_tx = batch_training_set(y,tx,batch_size)
        grad = grad_eval(batch_y, batch_tx, w)
        w = w - gamma * grad
        loss = loss_function(y, tx, w)


    return w,loss
