#!/usr/bin/env python
# coding: utf-8

# In[52]:


import pandas as pd
from pandas import read_csv
from collections import Counter
import numpy as np
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math

filename = r'C:\Users\praveen\project_adaboost\glass.csv'
df = read_csv(filename, header=None)

# data forming 
x = df
train, test = train_test_split(x, test_size = 0.2)
Y_train=train[9]
Y_test=test[9]
X_train=train.drop(9, axis = 1)
X_test=test.drop(9, axis = 1)
X_train=np.array(X_train)
X_test=np.array(X_test)
Y_train=np.array(Y_train)
Y_test=np.array(Y_test)

for i in range(len(Y_train)):
    if Y_train[i]<4:
        Y_train[i]=Y_train[i]-1
    if Y_train[i]>4:
        Y_train[i]=Y_train[i]-2
        
for i in range(len(Y_test)):
    if Y_test[i]<4:
        Y_test[i]=Y_test[i]-1
    if Y_test[i]>4:
        Y_test[i]=Y_test[i]-2

# one hot encoding of y train 
nb_classes = 6
targets = np.array(Y_train).reshape(-1)
y= np.eye(nb_classes)[targets]
labels_train=np.argmax(y,axis=1)

# one hot encoding of y train 
targets = np.array(Y_test).reshape(-1)
y_test= np.eye(nb_classes)[targets]
labels_test=np.argmax(y_test,axis=1)
(dimt1,)=(labels_test.shape)
(dim1,)=(labels_train.shape)


# In[53]:


def init_plot_settings():
  # Visualization Reference: https://towardsdatascience.com/a-simple-guide-to-beautiful-visualizations-in-python-f564e6b9d392
  sns.set_style('darkgrid') # darkgrid, white grid, dark, white and ticks
  plt.rc('axes', titlesize=18)     # fontsize of the axes title
  plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
  plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
  plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
  plt.rc('legend', fontsize=13)    # legend fontsize
  plt.rc('font', size=13)          # controls default text sizes
    
#error calculation for each and every class
def error_calc(ypred,labels_train,m):
    con=[]
    for i in range(6):
        error=0
        searchval = i
        ii = np.where(labels_train == searchval)[0]
        for i  in range(len(ii)):
            a=ii[i]
            if ypred[a]!=labels_train[a]:
                error=error+1
        con.append(error)
    return con


# In[60]:


alphas = []
alp=[]
training_errors = []
prediction_errors = []
prediction_testerrors = []
init_plot_settings()

Iterations = 20                          # Number of iterations is a hyperparameter
ERROR_TRAIN=np.zeros((6,Iterations))
Y_PRED=np.zeros((dim1,6))
Y_PRED_TEST=np.zeros((dimt1,6))
y_interim=np.zeros(dim1,dtype=int)

# Set weights initially
w_i = np.ones(len(labels_train)) * 1 / len(labels_train)  

#Adaboost Algorithm
for m in range(Iterations): 

    # Fit weak classifier and predict labels
    clf = DecisionTreeClassifier(max_depth =4 )     # Decision Tree depth is a hyper parameter
    clf.fit(X_train, labels_train, sample_weight = w_i)
    
    y_pred = clf.predict(X_train)          #  Predicting training values
    y_pred_test=clf.predict(X_test)        # Prediciting Testing Values
    
    Indicator=[]
    #Indicator function for calcualtion of error
    for i in range(len(labels_train)):
        if labels_train[i]!=(y_pred[i]):
            Indicator.append(1)
        else:
            Indicator.append(0)
    Indicator=np.array(Indicator)          # Indicator function
            
    #Total error Calculation
    error_m=np.sum(np.multiply(w_i,Indicator))/np.sum(w_i)

    #For training Errors graph 
    training_errors.append(error_m)
    
    #alpha Calculation
    alpha_m =np.log((1 - error_m) / error_m)
    alphas.append(alpha_m)
    
    WeightIndicator=[]
    for i in range(len(labels_train)):
        if Indicator[i]==1:
            WeightIndicator.append(1)
        else:
            WeightIndicator.append(-1)
            
    #Updation of weights  and learning rate is hyper parameter
    w_i=np.multiply(w_i,np.exp(np.multiply(0.25*alpha_m,Indicator)))   # WeightIndicator   
          
    # one hot encoding for predicated label of training
    targets = np.array(y_pred).reshape(-1)
    p_train= np.eye(nb_classes)[targets]
    
    Y_int=p_train*alpha_m
    Y_PRED=Y_int+Y_PRED
    y_pred_interim=(np.argmax((Y_PRED),axis=1))

    err=0
    for i in range(len(y_pred)):
        if y_pred_interim[i]!=labels_train[i]:
            err=err+1     
    prediction_errors.append(err)
    
    ERROR_TRAIN[:,m]=error_calc(y_pred_interim,labels_train,m)
        
    #one hot encoding for testing
    targets = np.array(y_pred_test).reshape(-1)
    p_test= np.eye(nb_classes)[targets]
    
    Y_int=p_test*alpha_m
    Y_PRED_TEST=Y_int+Y_PRED_TEST
    y_pred_test_interim=(np.argmax((Y_PRED_TEST),axis=1))

    err=0
    for i in range(len(y_pred_test_interim)):
        if y_pred_test_interim[i]!=labels_test[i]:
            err=err+1     
    prediction_testerrors.append(err)
        
#final predicition
y_pred_train=(np.argmax((Y_PRED),axis=1))
y_pred_test=(np.argmax((Y_PRED_TEST),axis=1))


#PLOTTING 
plt.figure(figsize=(10, 6))
plt.plot(prediction_errors,label='Training error')
plt.xlabel('Estimators')
plt.ylabel('Training Error')
plt.title('Training Error vs Number of Estimators')
plt.legend()


plt.figure(figsize=(10, 6))
ax = sns.lineplot(data=ERROR_TRAIN.T[:,:], linewidth=2.5)
ax.set(xlabel='Estimator', ylabel='Error/Loss value', title='Error/Loss vs estimator',xticks=[i for i in range(ERROR_TRAIN.T.shape[0])])
ax.legend(title='Estimator', title_fontsize = 13, loc=1)
plt.show()


# In[69]:


from sklearn.metrics import confusion_matrix

targets = np.array(y_pred_train).reshape(-1)
Y_PRED= np.eye(nb_classes)[targets]

targets = np.array(y_pred_test).reshape(-1)
Y_PRED_TEST= np.eye(nb_classes)[targets]

def compute_performance(y, y_pred):
    
    mean_sensitivity=[]
    mean_specificity=[]
    mean_precision=[]
    mean_f1_score=[]
    g_mean_avg=[]
    total_error = 0

    for i, y_val in enumerate(y):
        if not np.array_equal(y_val,y_pred[i]):
            total_error += 1
  
    acc_score = 1 - total_error/y.shape[0]

    acc = []
    cf = []
    precision = []
    recall = []


    for i in range(6):
#         print('Class_'+str(i+1))
        tn, fp, fn, tp = confusion_matrix(y[:,i], y_pred[:,i]).ravel()
        sensitivity = tp / (tp+fn)
        mean_sensitivity.append(sensitivity)
        specificity = tn / (tn+fp)
        mean_specificity.append(specificity)
        precision = tp / (tp + fp)
        mean_precision.append(precision)
        f1_score = 2*tp / (2*tp + fp + fn)
        mean_f1_score.append(f1_score)
        g_mean = (sensitivity * specificity) ** (1/2)
        g_mean_avg.append(g_mean)
        
        #for each invidual class performances
#         print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
#         print(f"Sensitivity : {sensitivity}")
#         print(f"Specificity : {specificity}")
#         print(f"Precision : {precision}")
#         print(f"F1-score : {f1_score}")
#         print(f"g-mean : {g_mean}")
    
     
    mean_sensitivity = [num for num in mean_sensitivity if num<len(labels_train)]
    mean_specificity = [num for num in mean_specificity if num<len(labels_train)]
    mean_precision = [num for num in mean_precision if num<len(labels_train)]
    mean_f1_score = [num for num in mean_f1_score if num<len(labels_train)]
    g_mean_avg = [num for num in g_mean_avg if num<len(labels_train)]
    
   
    sensitivity=sum(mean_sensitivity)/len(mean_sensitivity)
    specificity=sum(mean_specificity)/len(mean_specificity)
    precision=sum(mean_precision)/len(mean_precision)
    F1_score=sum(mean_f1_score)/len(mean_f1_score)
    g_mean=sum(g_mean_avg)/len(g_mean_avg)
    
    print(f"sensitivity : {sensitivity}")
    print(f"specificity : {specificity}")
    print(f"precision   : {precision}")
    print(f"F1_score : {F1_score}")
    print(f"G_mean : {g_mean}")
    
  


# In[70]:


print('The following are the training dataset performances')
compute_performance(y, Y_PRED)
print('The following are the testing dataset performances')
compute_performance(y_test,Y_PRED_TEST) 


# In[ ]:




