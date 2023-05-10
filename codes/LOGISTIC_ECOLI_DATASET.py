#!/usr/bin/env python
# coding: utf-8

# In[55]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


# In[48]:


file = pd.read_csv(r'C:\Users\praveen\project_adaboost\ecoli.csv')

df = file.values
train, test = train_test_split(df, test_size = 0.2)   
X_train= train[:,0:7]
Y_train = train[:,7]
X_test=test[:,0:7]
Y_test=test[:,7]

# one hot encoding of y train for multiclass 
targets1=[]
nb_classes = 8
targets = np.array(Y_train).reshape(-1)
for i in range(len(Y_train)):
    if Y_train[i]=='cp':
        targets1.append(0)
    if Y_train[i]=='im':
        targets1.append(1) 
    if Y_train[i]=='imU':
        targets1.append(2)
    if Y_train[i]=='om':
        targets1.append(3)       
    if Y_train[i]=='omL':
        targets1.append(4)
    if Y_train[i]=='pp':
        targets1.append(5) 
    if Y_train[i]=='imS':
        targets1.append(6)
    if Y_train[i]=='imL':
        targets1.append(7)  
y= np.eye(nb_classes)[targets1]

# one hot encoding of y test for multiclass
targets1=[]
targets = np.array(Y_test).reshape(-1)
for i in range(len(Y_test)):
    if Y_test[i]=='cp':
        targets1.append(0)
    if Y_test[i]=='im':
        targets1.append(1) 
    if Y_test[i]=='imU':
        targets1.append(2)
    if Y_test[i]=='om':
        targets1.append(3)       
    if Y_test[i]=='omL':
        targets1.append(4)
    if Y_test[i]=='pp':
        targets1.append(5) 
    if Y_test[i]=='imS':
        targets1.append(6)
    if Y_test[i]=='imL':
        targets1.append(7)
y_test= np.eye(nb_classes)[targets1]
labels_train=np.argmax(y,axis=1)
lables_test=np.argmax(y_test,axis=1)


# In[49]:


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
    for i in range(8):
        error=0
        searchval = i
        ii = np.where(labels_train == searchval)[0]
        for i  in range(len(ii)):
            a=ii[i]
            if ypred[a]!=labels_train[a]:
                error=error+1
        con.append(error)
    return con


# In[50]:


init_plot_settings()
trainingerror=[]
ERROR_TRAIN=np.zeros((8,20))
for i in range(20):
    clf = LogisticRegression(max_iter=i)
    clf.fit(X_train,labels_train)
    y_pred_train= clf.predict(X_train)
    y_pred_test=clf.predict(X_test)
    accuracy = accuracy_score(labels_train, y_pred_train)
#     print('Accuracy Score',(1-accuracy)*100)
    trainingerror.append((1-accuracy)*100)
    
    ERROR_TRAIN[:,i]=error_calc(y_pred_train,labels_train,i)
    


# In[51]:


plt.figure(figsize=(10, 6))
ax = sns.lineplot(data=ERROR_TRAIN.T[:,:], linewidth=2.5)
ax.set(xlabel='Estimator', ylabel='Error/Loss value', title='Error/Loss vs estimator',xticks=[i for i in range(ERROR_TRAIN.T.shape[0])])
ax.legend(title='Estimator', title_fontsize = 13, loc=1)
plt.show()


# In[52]:


from sklearn.metrics import confusion_matrix
nb_classes=8
targets = np.array(lables_test).reshape(-1)
y_test= np.eye(nb_classes)[targets]
targets = np.array(y_pred_test).reshape(-1)
y_pred_test= np.eye(nb_classes)[targets]
targets = np.array(y_pred_train).reshape(-1)
y_pred_train= np.eye(nb_classes)[targets]


# In[53]:


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


    for i in range(4):
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
    


# In[54]:


print('The following are the training dataset performances')
compute_performance(y,y_pred_train )
print('The following are the testing dataset performances')
compute_performance(y_test,y_pred_test)


# In[ ]:




