import numpy as np
import matplotlib.pyplot as plt
from template.helpers import *


def ridge_regression(y, tx, lambda_):
    #split_y = y[:100]
    #split_tx = tx[:100]
    split_y = y
    split_tx = tx


    a = split_tx.T.dot(split_tx)
    #print('A : {a}'.format(a=a))

    b = 2*len(split_y)* lambda_ * np.identity(len(split_tx[0]))
    #print('B : {b}'.format(b=b))


    c = split_tx.T.dot(split_y)
    #print('C : {c}'.format(c=c))

    w = np.linalg.solve((a+b), c)
    loss = loss_function(y,tx,w)


    return w,loss
