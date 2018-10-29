import numpy as np
import matplotlib.pyplot as plt
from template.helpers import *
from least_squares_GD import *
from logistic_regression import *
from reg_logistic_regression import *

pathTrain = "./template/train.csv"
pathTest = "./template/test.csv"


yb,input_data,ids = load_csv_data_logistic(pathTrain)
yb_test,input_data_test,ids_test = load_csv_data_logistic(pathTest)
#yb -> y vector : it's the real output
#input_data -> x parameters
#ids -> id of each data row

np.putmask(input_data, input_data == -999, 0)

input_data = standardize(input_data)
input_data_test = standardize(input_data_test)

gamma = 0.01
max_iters = 7000


#tx = np.c_[np.ones(len(yb)), input_data]
tx = build_poly(input_data, 2)
tx_test = build_poly(input_data_test, 2)

w_initial = np.zeros(len(tx[0]))

wLR, loss = logistic_regression_GD(yb, tx, w_initial, gamma, max_iters)

y_pred = predict_labels_logistic(wLR, tx)
percent = comparePredict_correct(yb, y_pred)
y_pred_test = predict_labels_logistic_test(wLR,tx_test)
create_csv_submission(ids_test,y_pred_test,'optimized_logisticGD.csv')
print(percent)
