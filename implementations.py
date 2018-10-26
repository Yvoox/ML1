import numpy as np
import matplotlib.pyplot as plt
from template.helpers import *
from least_squares_GD import *
from logistic_regression import *
from reg_logistic_regression import *

pathTrain = "./template/train.csv"
pathTest = "./template/test.csv"


yb,input_data,ids = load_csv_data_logistic(pathTrain)
#yb -> y vector : it's the real output
#input_data -> x parameters
#ids -> id of each data row

#np.putmask(input_data, input_data == -999, 0)

input_data = standardize(input_data)

w_initial = np.zeros(len(input_data[0])+1)
gamma = 0.001
lambda_ = 0.1
max_iters = 100
threshold = 1e-5

tx = np.c_[np.ones(len(yb)), input_data]

wLR = logistic_regression_GD(yb, tx, w_initial, gamma, max_iters, threshold)

y_pred = predict_labels_logistic(wLR, tx)
percent = comparePredict_correct(yb, y_pred)

#create_csv_submission(ids, 1 / (1 + np.exp(- np.dot(tx, wLR))), yb, 'monvul.csv')

print(percent)
