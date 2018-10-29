import numpy as np
import matplotlib.pyplot as plt
from template.helpers import *
from least_squares_SGD import *
from least_squares import *
from ridge_regression import *
from crossValidation import *
from least_squares_GD import *
from logistic_regression import *
from reg_logistic_regression import *

import random





pathTrain = "./template/train.csv"
pathTest = "./template/test.csv"

#STANDARD DATA SET

yb,input_data,ids = load_csv_data(pathTrain)
#yb_test,input_data_test,ids_test = load_csv_data(pathTest)

#LOGISTIC DATA SET (YB INTERVAL 0,1)

yb_log,input_data_log,ids_log = load_csv_data_logistic(pathTrain)
#yb_test_log,input_data_test_log,ids_test_log = load_csv_data_logistic(pathTest)


#yb -> y vector : it's the real output
#input_data -> x parameters
#ids -> id of each data row

#Clean -999 values
np.putmask(input_data, input_data==-999, 0)
np.putmask(input_data_log, input_data_log==-999, 0)
#np.putmask(input_data_test, input_data_test==-999, 0)
#np.putmask(input_data_test_log, input_data_test_log==-999, 0)








#DataSet Standardization

input_data = standardize(input_data)
input_data_log = standardize(input_data_log)
#input_data_test = standardize(input_data_test)
#input_data_test_log = standardize(input_data_test_log)


#Add 1 before data

#tx = np.c_[np.ones(len(yb)), input_data]
#tx_log = np.c_[np.ones(len(yb)), input_data]
#tx_test = np.c_[np.ones(len(yb_test)), input_data_test]

#Or build a polynomial

tx = build_poly(input_data, 2)
tx_log = build_poly(input_data_log, 2)
#tx_test = build_poly(input_data_test, 2)
#tx_test_log = build_poly(input_data_test_log, 2)


#Constants definition

gamma = 0.01
gamma_log = 0.01
max_iters = 5000
max_iters_log = 7000
w_initial = np.zeros(len(tx[0]))
w_initial_log = np.zeros(len(tx_log[0]))





print('Data loaded & prepared...')




#---------------- CORRELATION VISUALIZATION  -------------
#for it in range(30):
#    print ('column nb :' + repr(it)+ ' corr:'+ repr(np.corrcoef(yb,column(tx,it))))



#----------------  GENETIC ALGORITHM  -------------------

def randomIndividuLeastSquares(nbTrain,yb,tx):
    index = np.random.choice(yb.shape[0], nbTrain, replace=False) #create nbTrain random idex
    wLS,loss = least_squares(yb[index],tx[index]) #LS with nbTrain sample
    y_pred = predict_labels(wLS,tx) #predict to prepare a comparative value
    percent = comparePredict(yb,y_pred) #comparative function
    return [wLS,percent]

def repro(p,m):
    cWLS = []
    pWLS = p[0] #take wLS from ind1
    mWLS = m[0] #take wLS from ind2
    rand = random.randint(0,101)

    for i in range (len(pWLS)):
        if i % 2 == 0:
            cWLS.append(pWLS[i])
        else:
            cWLS.append(mWLS[i])
    if rand > 95:
        print('Mutation appenned')
        pos =   random.randint(0,len(cWLS)-1)
        cWLS[pos] =  random.randint(-100,100)

    y_pred = predict_labels(cWLS,tx) #predict to prepare a comparative value
    percent = comparePredict(yb,y_pred) #comparative value
    return [cWLS,percent]

#Algo start

#population generation
pop = []
for i in range(50): #number of individus
    pop.append(randomIndividuLeastSquares(100,yb,tx)) #number of data in subdataTraining

sortedPop = sorted(pop,key=lambda x: x[1])


#starting Genetic
cptGen = 1
while len(sortedPop)!=1 :
    print('Generation number : '+repr(cptGen))
    child = []
    sortedPop = sorted(sortedPop,key=lambda x: x[1])

    for i in range(int(len(sortedPop)/2)):
        p = sortedPop[random.randint(0,len(sortedPop)-1)]
        m = sortedPop[random.randint(0,len(sortedPop)-1)]
        child.append(repro(p,m))
    sortedPop = np.copy(child)
    cptGen = cptGen +1

print(sortedPop[0])

#End genetic algorithm SGD

#LEAST SQUARE GD TEST
#wGD,lossGD = regression_GD(yb, tx, w_initial, gamma, max_iters)
#y_pred = predict_labels(wGD,tx)
#percent = comparePredict(yb,y_pred)
#print('Positive percent with standard tx : '+repr(percent)+' with loss: '+repr(lossGD))
#y_pred_test = predict_labels(wGD,tx_test)
#create_csv_submission(ids_test,y_pred_test,'least_squares_gd.csv')


#LEAST SQUARES SGD TEST
#wSGD,lossSGD = least_squares_SGD(yb,tx,w_initial,max_iters,gamma)
#s = 'SGD : w : ' + repr(wSGD) + ' - Loss: ' + repr(lossSGD)
#print(s)
#y_pred = predict_labels(wSGD,tx)
#percent = comparePredict(yb,y_pred)
#print('Error percent with standard tx : '+repr(percent)+'with loss: '+repr(lossSGD))
#y_pred_test = predict_labels(wSGD,tx_test)
#create_csv_submission(ids_test,y_pred_test,'least_squares_sgd.csv')



#LEAST SQUARES TEST
#wLS,loss = least_squares(yb,tx)
#y_pred = predict_labels(wLS,tx)
#percent = comparePredict(yb,y_pred)
#print('Error percent with standard tx : '+repr(percent)+'with loss : '+repr(loss))

#y_pred_test = predict_labels(wLS,tx_test)

#create_csv_submission(ids_test,y_pred_test,'least_squares_opti.csv')


#RIDGE REGRESSION TEST
#wRD,loss = ridge_regression(yb,tx,1)
#y_pred = predict_labels(wRD,tx)
#percent = comparePredict(yb,y_pred)

#print('Error percent with standard tx : '+repr(percent)+'with loss : '+repr(loss))
#y_pred_test = predict_labels(wRD,tx_test)
#create_csv_submission(ids_test,y_pred_test,'ridge_regression_opti.csv')



#LOGISTIC REGRESSION GD TEST
#wLR, loss = logistic_regression_GD(yb_log, tx_log, w_initial_log, gamma_log, max_iters_log)

#y_pred = predict_labels_logistic(wLR, tx_log)
#percent = comparePredict_correct(yb_log, y_pred)
#y_pred_test = predict_labels_logistic_test(wLR,tx_test_log)
#print('Positive percent with logistic tx : '+repr(percent)+'with loss : '+repr(loss))
#create_csv_submission(ids_test_log,y_pred_test,'optimized_logisticGD.csv')


#REG LOGISTIC REGRESSION TEST
#wRLR,lossRLR = reg_logistic_regression_GD(yb_log, tx_log, w_initial_log, gamma_log, max_iters_log, 0.1)
#y_pred = predict_labels_logistic(wRLR, tx_log)
#percent = comparePredict_correct(yb_log, y_pred)
#print('Positive percent with logistic tx : '+repr(percent)+'with loss : '+repr(lossRLR))


#-----------------  CROSS VALIDATION  Example with Least Squares--------------
#third argument list
#1- Least Squares
#2- Least Squares GD
#3- Least Squares SGD
#4- Ridge Regression
#YOU HAVE TO PASS tx_log & yb_log FOR LOGISTIC ET REG LOGISTIC
#5- Logistic Regression
#6- Reg Logistic Regression


#crossValidation(tx,yb,1,len(input_data[0])+1)
#crossValidation(tx_log,yb_log,6,len(input_data_log[0])+1)
