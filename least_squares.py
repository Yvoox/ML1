import numpy as np
import matplotlib.pyplot as plt
from template.helpers import *


def least_squares(y, tx):


    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = loss_function(y,tx,w)
    return w,loss
