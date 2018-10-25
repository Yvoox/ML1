import numpy as np
import matplotlib.pyplot as plt
from template.helpers import *
from logistic_regression import *
from reg_logistic_regression import *

pathTrain = "./template/train.csv"
pathTest = "./template/test.csv"


yb,input_data,ids = load_csv_data_logistic(pathTrain)
#yb -> y vector : it's the real output
#input_data -> x parameters
#ids -> id of each data row

input_data = standardize(input_data)

w_initial = np.zeros(len(input_data[0])+1)
gamma = 0.1
lambda_ = 0.1
max_iters = 1000
threshold = 1e-8

tx = np.c_[np.ones(len(yb)), input_data]

wLR = logistic_regression_GD(yb, tx, w_initial, gamma, max_iters, threshold)

y_pred = predict_labels_logistic(wLR, tx)
percent = comparePredict(yb, y_pred)

print(percent)
