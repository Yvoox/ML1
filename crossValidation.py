import numpy as np
from template.helpers import *
from least_squares import *
from ridge_regression import *
from least_squares_SGD import *


import random



#-----------------  CROSS VALIDATION  --------------
def crossValidation(tx,yb,arg,data_len):
    random_w = np.random.random(data_len)
    gamma = 0.7
    max_iters = 50
    #Holdout Method for least squares
    tx_train,tx_test,yb_train,yb_test = splitDataHD(tx,yb,125000)

    if(arg==1):
        print('LEAST SQUARE CROSS VALIDATION PROCESSING')
        wLS = least_squares(yb_train,tx_train)
        y_pred = predict_labels(wLS,tx_test)
        percent = comparePredict(yb_test,y_pred)
    if(arg==3):
        print('LEAST SQUARE SGD CROSS VALIDATION PROCESSING')
        wSGD,lossSGD = least_squares_SGD(yb_train,tx_train,random_w,max_iters,gamma)
        y_pred = predict_labels(wSGD,tx_test)
        percent = comparePredict(yb_test,y_pred)
    if(arg==4):
        print('RIDGE REGRESSION CROSS VALIDATION PROCESSING')
        wRD = ridge_regression(yb_train,tx_train,1)
        y_pred = predict_labels(wRD,tx_test)
        percent = comparePredict(yb_test,y_pred)

    print('Error percent with Holdout Method : '+repr(percent))

    #K-Fold Cross Validation for least squares
    k = 10
    result = 0
    tx_list,yb_list = splitDataKFold(tx,yb,k)
    if(arg==1):
        for i in range(k-1, 0, -1):
            wLS = least_squares(yb_list[i-1],tx_list[i-1])
            y_pred = predict_labels(wLS,tx_list[i])
            percent = comparePredict(yb_list[i],y_pred)
            result = result + percent
    if(arg==3):
        for i in range(k-1, 0, -1):
            wSGD,lossSGD = least_squares_SGD(yb_list[i-1],tx_list[i-1],random_w,max_iters,gamma)
            y_pred = predict_labels(wSGD,tx_list[i])
            percent = comparePredict(yb_list[i],y_pred)
            result = result + percent
    if(arg==4):
        for i in range(k-1, 0, -1):
            wRD = ridge_regression(yb_list[i-1],tx_list[i-1],1)
            y_pred = predict_labels(wRD,tx_list[i])
            percent = comparePredict(yb_list[i],y_pred)
            result = result + percent

    print('Error percent with K-Fold Cross Validation Method : '+repr(result/k))
