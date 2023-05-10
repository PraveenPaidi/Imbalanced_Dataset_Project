#!/usr/bin/env python
# coding: utf-8

# In[12]:


from google.colab import drive
drive.mount('/content/drive')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import classification_report


# In[13]:


def read_data(file):
  '''
  Function to read data
  Data source: 

  '''
  df = pd.read_csv(file, header=None)
  num_col = len(df. columns) - 1
  X = df.iloc[:, 0:num_col]
  y = df.iloc[:,num_col]
  y = y.values.reshape(-1,1)
  encoder = OneHotEncoder(sparse=False)
  y_OneHot = encoder.fit_transform(y)
  return X, y_OneHot


# In[14]:


def data_standardization(X):
  '''
  Function to standardize the data using StandardScaler
  X: input data
  '''
  sc = StandardScaler()
  X = sc.fit_transform(X)
  return X


# In[15]:


def init_plot_settings():
  # Visualization Reference: https://towardsdatascience.com/a-simple-guide-to-beautiful-visualizations-in-python-f564e6b9d392
  sns.set_style('darkgrid') # darkgrid, white grid, dark, white and ticks
  plt.rc('axes', titlesize=18)     # fontsize of the axes title
  plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
  plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
  plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
  plt.rc('legend', fontsize=13)    # legend fontsize
  plt.rc('font', size=13)          # controls default text sizes


# In[16]:


def plot_loss(error):
  error = np.array(error)
  plt.figure(figsize=(10, 6))
  ax = sns.lineplot(data=error[:,:], linewidth=2.5)
  ax.set(xlabel='Estimator', ylabel='Error/Loss value', title='Error/Loss vs estimator',xticks=[i for i in range(error.shape[0])])
  ax.legend(title='Estimator', title_fontsize = 13, loc=1)
  plt.show()


# In[17]:


def Gradient_Boosting(X_train, y_train, n_estimator = 10, max_depth = 2, lr = 0.01):
  gradient_boosting_estimator = []
  y_train_org = y_train.copy()
  train_error = []
  for i in range(n_estimator):
    clf = DecisionTreeRegressor(max_depth=max_depth, random_state=1)
    model = clf.fit(X_train, y_train)

    y_pred = model.predict(X_train)
    loss_grad = y_train - y_pred
    train_error.append(sum(abs(loss_grad)))
    
    y_train = y_train - lr * loss_grad

    gradient_boosting_estimator.append((lr, model))

 
  plot_loss(train_error)

  return gradient_boosting_estimator


# In[18]:


def predict(X, estimator, classes):
  pred_vals = np.zeros((X.shape[0], classes))
  for val in estimator:
    pred_vals += val[0] * val[1].predict(X)

  y_pred = np.zeros((X.shape[0], classes))

  for i, pred in enumerate(pred_vals):
    max_idx = np.argmax(pred)
    y_pred[i][max_idx] = 1

  return y_pred 


# In[19]:


def Decision_tree(X, y):
  clf = DecisionTreeClassifier(max_depth = 1)
  clf.fit(X,y)
  return clf


# In[20]:


def compute_preformance(y, y_pred, classes):

  total_error = 0

  for i, y_val in enumerate(y):
    if not np.array_equal(y_val,y_pred[i]):
      total_error += 1
  
  acc_score = 1 - total_error/y.shape[0]

  acc = []
  cf = []
  precision = []
  recall = []
  
  target_names = ['Class'+str(i+1) for i in range(classes)]

  for i in range(y.shape[1]):
    print('Class_'+str(i+1))
    tn, fp, fn, tp = confusion_matrix(y[:,i], y_pred[:,i]).ravel()
    sensitivity = tp / (tp+fn)
    specificity = tn / (tn+fp)
    precision = tp / (tp + fp)
    f1_score = 2*tp / (2*tp + fp + fn)
    g_mean = (sensitivity * specificity) ** (1/2)
    print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
    print(f"Sensitivity : {sensitivity}")
    print(f"Specificity : {specificity}")
    print(f"Precision : {precision}")
    print(f"F1-score : {f1_score}")
    print(f"g-mean : {g_mean}")

    


# In[21]:


file_path = '/content/drive/MyDrive/SML Project/ecoli.csv'
# file_path = '/content/drive/MyDrive/SML Project/glass.csv'
# file_path = '/content/drive/MyDrive/SML Project/wifi.csv'
X, y = read_data(file_path)
classes = y.shape[1]
init_plot_settings()


# In[21]:





# In[22]:


# Data creation
X = data_standardization(X)
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[23]:


gradient_boosting_estimator = Gradient_Boosting(X_train, y_train, n_estimator = 20, max_depth = 4, lr = 0.01)


# In[24]:


y_pred_train = predict(X_train, gradient_boosting_estimator, classes)
y_pred_test = predict(X_test, gradient_boosting_estimator, classes)
print(f"Train:")
compute_preformance(y_train, y_pred_train, classes)

print(f"Test:")
compute_preformance(y_test, y_pred_test, classes)


# In[24]:





# In[27]:


# Decision Tree Predictions
dt = Decision_tree(X_train, y_train)
y_pred_train = dt.predict(X_train)
y_pred_test = dt.predict(X_test)
print(f"Train:")
compute_preformance(y_train, y_pred_train, classes)

print(f"Test:")
compute_preformance(y_test, y_pred_test, classes)

