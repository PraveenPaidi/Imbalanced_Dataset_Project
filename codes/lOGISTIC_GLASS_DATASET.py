#!/usr/bin/env python
# coding: utf-8

# In[187]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[188]:


file = pd.read_csv (r'C:\Users\praveen\project_adaboost\glass.csv')

df = file.values
train, test = train_test_split(df, test_size = 0.2,random_state=1)   
X_train= train[:,0:9]
Y_train = train[:,9]
X_test=test[:,0:9]
Y_test=test[:,9]

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


# In[189]:


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


# In[190]:


init_plot_settings()
trainingerror=[]
ERROR_TRAIN=np.zeros((6,150))
for i in range(150):
    clf = LogisticRegression(max_iter=i)
    clf.fit(X_train,Y_train)
    y_pred_train= clf.predict(X_train)
    y_pred_test=clf.predict(X_test)
    accuracy = accuracy_score(Y_train, y_pred_train)
#     print('Accuracy Score',(1-accuracy)*100)
    trainingerror.append((1-accuracy)*100)
    
    ERROR_TRAIN[:,i]=error_calc(y_pred_train,Y_train,i)
    


# In[191]:


plt.figure(figsize=(10, 6))
ax = sns.lineplot(data=ERROR_TRAIN.T[:,:], linewidth=2.5)
ax.set(xlabel='Estimator', ylabel='Error/Loss value', title='Error/Loss vs estimator')
ax.legend(title='Estimator', title_fontsize = 13)
plt.show()


# In[192]:


nb_classes=6
targets = np.array(y_pred_test,dtype=int).reshape(-1)
y_pred_test= np.eye(nb_classes)[targets]
targets = np.array(y_pred_train,dtype=int).reshape(-1)
y_pred_train= np.eye(nb_classes)[targets]
targets = np.array(Y_train,dtype=int).reshape(-1)
Y_train= np.eye(nb_classes)[targets]
targets = np.array(Y_test,dtype=int).reshape(-1)
Y_test= np.eye(nb_classes)[targets]


# In[193]:


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
        
    mean_sensitivity = [num for num in mean_sensitivity if num<len(Y_train)]
    mean_specificity = [num for num in mean_specificity if num<len(Y_train)]
    mean_precision = [num for num in mean_precision if num<len(Y_train)]
    mean_f1_score = [num for num in mean_f1_score if num<len(Y_train)]
    g_mean_avg = [num for num in g_mean_avg if num<len(Y_train)]

    sensitivity=sum(mean_sensitivity)/len(mean_sensitivity)
    specificity=sum(mean_specificity)/len(mean_specificity)
    precision=sum(mean_precision)/len(mean_precision)
    F1_score=sum(mean_f1_score)/len(mean_f1_score)
    g_mean=sum(g_mean_avg)/len(g_mean_avg)

    print(f"sensitivity : {sensitivity}")
    print(f"specificity : {specificity}")
    print(f"precision   : {precision}")
    print(f"F1_score    : {F1_score}")
    print(f"G_mean      : {g_mean}")
    


# In[194]:


print('The following are the training dataset performances')
compute_performance(Y_train,y_pred_train )
print('The following are the testing dataset performances')
compute_performance(Y_test,y_pred_test)


# In[ ]:




