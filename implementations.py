import numpy as np
import matplotlib.pyplot as plt
from template.helpers import *
from least_squares_GD import *

pathTrain = "./template/train.csv"
pathTest = "./template/test.csv"


yb,input_data,ids = load_csv_data(pathTrain)
#yb -> y vector : it's the real output
#input_data -> x parameters
#ids -> id of each data row

input_data = standardize(input_data)

random_w = np.random.random(len(input_data[0])+1)
gamma = 0.7
max_iters = 50

tx = np.c_[np.ones(len(yb)), input_data]

lossGD, wGD = gradient_descent(yb, tx, random_w, max_iters, gamma)

y_pred = predict_labels(wGD, tx)
percent = comparePredict(yb, y_pred)

print(repr(percent))

input()
