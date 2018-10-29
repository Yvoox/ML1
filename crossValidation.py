import numpy as np
from template.helpers import *
from least_squares import *
from ridge_regression import *
from least_squares_SGD import *
from least_squares_GD import *
from logistic_regression import *
from reg_logistic_regression import *


import random



#-----------------  CROSS VALIDATION  --------------
def crossValidation(tx,yb,arg,data_len):
    w_initial = np.zeros(len(tx[0]))
    gamma = 0.7
    max_iters = 50
    #Holdout Method for least squares
    tx_train,tx_test,yb_train,yb_test = splitDataHD(tx,yb,125000)

    if(arg==1):
        print('LEAST SQUARE CROSS VALIDATION PROCESSING')
        wLS,loss = least_squares(yb_train,tx_train)
        y_pred = predict_labels(wLS,tx_test)
        percent = comparePredict(yb_test,y_pred)
    if(arg==2):
        print('LEAST SQUARE GD CROSS VALIDATION PROCESSING')
        wGD,lossGD = regression_GD(yb, tx, w_initial, 0.01, 5000)
        y_pred = predict_labels(wGD,tx)
        percent = comparePredict(yb,y_pred)
    if(arg==3):
        print('LEAST SQUARE SGD CROSS VALIDATION PROCESSING')
        wSGD,lossSGD = least_squares_SGD(yb_train,tx_train,w_initial,5000,0.01)
        y_pred = predict_labels(wSGD,tx_test)
        percent = comparePredict(yb_test,y_pred)
    if(arg==4):
        print('RIDGE REGRESSION CROSS VALIDATION PROCESSING')
        wRD,loss = ridge_regression(yb_train,tx_train,1)
        y_pred = predict_labels(wRD,tx_test)
        percent = comparePredict(yb_test,y_pred)
    if(arg==5):
        print('LOGISTIC REGRESSION CROSS VALIDATION PROCESSING')
        wLR, loss = logistic_regression_GD(yb_train, tx_train, w_initial, 0.01, 7000)
        y_pred = predict_labels_logistic(wLR, tx_test)
        percent = comparePredict_correct(yb_test, y_pred)
    if(arg==6):
        print('REG LOGISTIC REGRESSION CROSS VALIDATION PROCESSING')
        wRLR,lossRLR = reg_logistic_regression_GD(yb_train, tx_train, w_initial, 0.01, 7000, 0.1)
        y_pred = predict_labels_logistic(wRLR, tx_test)
        percent = comparePredict_correct(yb_test, y_pred)

    print('Error percent with Holdout Method : '+repr(percent))

    #K-Fold Cross Validation for least squares
    k = 10
    result = 0
    tx_list,yb_list = splitDataKFold(tx,yb,k)
    if(arg==1):
        for i in range(k-1, 0, -1):
            wLS,loss = least_squares(yb_list[i-1],tx_list[i-1])
            y_pred = predict_labels(wLS,tx_list[i])
            percent = comparePredict(yb_list[i],y_pred)
            result = result + percent
    if(arg==2):
        for i in range(k-1, 0, -1):
            wGD,lossGD = regression_GD(yb, tx, w_initial, 0.01, 5000)
            y_pred = predict_labels(wGD,tx)
            percent = comparePredict(yb,y_pred)
            result = result + percent
    if(arg==3):
        for i in range(k-1, 0, -1):
            wSGD,lossSGD = least_squares_SGD(yb_list[i-1],tx_list[i-1],w_initial,5000,0.01)
            y_pred = predict_labels(wSGD,tx_list[i])
            percent = comparePredict(yb_list[i],y_pred)
            result = result + percent
    if(arg==4):
        for i in range(k-1, 0, -1):
            wRD,lossRD = ridge_regression(yb_list[i-1],tx_list[i-1],1)
            y_pred = predict_labels(wRD,tx_list[i])
            percent = comparePredict(yb_list[i],y_pred)
            result = result + percent
    if(arg==5):
        for i in range(k-1, 0, -1):
            wLR,lossLR = logistic_regression_GD(yb_list[i-1],tx_list[i-1],w_initial, 0.01, 7000)
            y_pred = predict_labels(wLR,tx_list[i])
            percent = comparePredict(yb_list[i],y_pred)
            result = result + percent
    if(arg==6):
        for i in range(k-1, 0, -1):
            wRLR,lossRLR = reg_logistic_regression_GD(yb_list[i-1],tx_list[i-1],w_initial, 0.01, 7000, 0.1)
            y_pred = predict_labels(wRLR,tx_list[i])
            percent = comparePredict(yb_list[i],y_pred)
            result = result + percent

    print('Error percent with K-Fold Cross Validation Method : '+repr(result/k-1))
