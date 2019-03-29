
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from numpy.linalg import inv
from sklearn import model_selection
from matplotlib import pyplot as plt
from data import pcaNames, smallPca, linearRegressionData, normalizedRegressionData


# In[4]:


# Load in the ozone data
# df = pd.read_csv('ozone.csv')


# In[5]:


# Normalize the data to achieve standard devation 1 and mean 0
# norm_df = (df - df.mean()) / df.std()


# In[64]:
X = normalizedRegressionData[:, list(range(len(normalizedRegressionData[0]) - 1))]
y = normalizedRegressionData[:, -1].squeeze()


num_lambdas = 20
lambdas = np.linspace(start=1e-6, stop=100, num=num_lambdas)
lambda_error = np.empty(len(lambdas))
ws = np.empty((num_lambdas, X.shape[1]))
for idx, lamb in enumerate(lambdas):
    k = 0
    K = 10
    CV = model_selection.KFold(n_splits=K, shuffle=True)
    test_error = np.empty(K)
    fold_ws = np.empty((K, ws.shape[1]))
    for train_idx, test_idx in CV.split(X):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        # Optimal weights with regularization parameter lambda are given by (X.T@X + lamb@I)^-1 @ (X.T@y)
        w = inv(X_train.T @ X_train + lamb * np.identity(X_train.shape[1])) @ (X_train.T @ y_train)
        fold_ws[k] = w
        y_pred = X_test @ w
        # MSE
        test_error[k] = ((y_pred - y_test)**2).mean()
        k += 1
    avg_error = test_error.mean()
    ws[idx] = fold_ws.mean(axis=0)
    lambda_error[idx] = avg_error


# In[65]:


#get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(lambdas, lambda_error)


# In[59]:


# plt.plot(lambdas, ws)
plt.show()

