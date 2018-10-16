import numpy as np
import matplotlib.pyplot as plt

def least_squares(y, tx):
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    return np.linalg.solve(a, b)
