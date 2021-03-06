import numpy as np
import matplotlib.pyplot as plt
from template.helpers import *
from least_squares_SGD import *
from least_squares import *
from ridge_regression import *
from crossValidation import *

import random





pathTrain = "./template/train.csv"
pathTest = "./template/test.csv"


yb,input_data,ids = load_csv_data(pathTrain)
#yb_test,input_data_test,ids_test = load_csv_data(pathTest)


#yb -> y vector : it's the real output
#input_data -> x parameters
#ids -> id of each data row

#Clean -999 values
np.putmask(input_data, input_data==-999, 0)
#np.putmask(input_data_test, input_data_test==-999, 0)










input_data = standardize(input_data)


#input_data_test = standardize(input_data_test)

random_w = np.random.random(len(input_data[0])+1)
gamma = 0.7
max_iters = 50

tx = np.c_[np.ones(len(yb)), input_data]

#tx_test = np.c_[np.ones(len(yb_test)), input_data_test]



#---------------- CORRELATION VISUALIZATION  -------------
#for it in range(30):
#    print ('column nb :' + repr(it)+ ' corr:'+ repr(np.corrcoef(yb,column(tx,it))))

print('Data loaded & prepared...')


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
#pop = []
#for i in range(50): #number of individus
#    pop.append(randomIndividuLeastSquares(100,yb,tx)) #number of data in subdataTraining

#sortedPop = sorted(pop,key=lambda x: x[1])


#starting Genetic
#cptGen = 1
#while len(sortedPop)!=1 :
#    print('Generation number : '+repr(cptGen))
#    child = []
#    sortedPop = sorted(sortedPop,key=lambda x: x[1])

#    for i in range(int(len(sortedPop)/2)):
#        p = sortedPop[random.randint(0,len(sortedPop)-1)]
#        m = sortedPop[random.randint(0,len(sortedPop)-1)]
#        child.append(repro(p,m))
#    sortedPop = np.copy(child)
#    cptGen = cptGen +1

#print(sortedPop[0])

#End genetic algorithm SGD



#LEAST SQUARES SGD TEST
#wSGD,lossSGD = least_squares_SGD(yb,tx,random_w,max_iters,gamma)
#s = 'SGD : w : ' + repr(wSGD) + ' - Loss: ' + repr(lossSGD)
#print(s)
#y_pred = predict_labels(wSGD,tx)
#percent = comparePredict(yb,y_pred)
#print('Percent with standard tx : '+repr(percent)+'with loss: '+repr(lossSGD))
#y_pred_test = predict_labels(wSGD,tx_test)
#create_csv_submission(ids_test,y_pred_test,'least_squares_sgd.csv')



#LEAST SQUARES TEST
#wLS,loss = least_squares(yb,tx)
#y_pred = predict_labels(wLS,tx)
#percent = comparePredict(yb,y_pred)
#print('Percent with standard tx : '+repr(percent)+'with loss : '+repr(loss))

#y_pred_test = predict_labels(wLS,tx_test)

#create_csv_submission(ids_test,y_pred_test,'least_squares_opti.csv')


#RIDGE REGRESSION TEST
#wRD,loss = ridge_regression(yb,tx,1)
#y_pred = predict_labels(wRD,tx)
#percent = comparePredict(yb,y_pred)

#print('Percent with standard tx : '+repr(percent)+'with loss : '+repr(loss))
#y_pred_test = predict_labels(wRD,tx_test)
#create_csv_submission(ids_test,y_pred_test,'ridge_regression_opti.csv')

#-----------------  CROSS VALIDATION  Example with Least Squares--------------
#third argument list
#1- Least Squares
#2- Least Squares GD
#3- Least Squares SGD
#4- Ridge Regression
#5- Logistic Regression
#6- Reg Logistic Regression


#crossValidation(tx,yb,3,len(input_data[0])+1)


print('CSV created... Operation done')


input()
