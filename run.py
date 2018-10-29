import numpy as np
import matplotlib.pyplot as plt
from template.helpers import *
from least_squares import *
from crossValidation import *


import random


pathTrain = "./template/train.csv"
pathTest = "./template/test.csv"


yb_log,input_data_log,ids_log = load_csv_data_logistic(pathTrain)
yb_test_log,input_data_test_log,ids_test_log = load_csv_data_logistic(pathTest)

np.putmask(input_data_log, input_data_log==-999, 0)
np.putmask(input_data_test_log, input_data_test_log==-999, 0)

input_data_log = standardize(input_data_log)
input_data_test_log = standardize(input_data_test_log)

tx_log = build_poly(input_data_log, 2)
tx_test_log = build_poly(input_data_test_log, 2)

gamma_log = 0.01
max_iters_log = 7000
w_initial_log = np.zeros(len(tx_log[0]))


print('Data loaded & prepared...')

wLR, loss = logistic_regression_GD(yb_log, tx_log, w_initial_log, gamma_log, max_iters_log)

y_pred = predict_labels_logistic(wLR, tx_log)
percent = comparePredict_correct(yb_log, y_pred)
y_pred_test = predict_labels_logistic_test(wLR,tx_test_log)
print('Positive percent with logistic tx : '+repr(percent)+' with loss : '+repr(loss))
create_csv_submission(ids_test_log,y_pred_test,'optimized_logisticGD.csv')
