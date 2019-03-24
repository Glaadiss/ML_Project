import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from data import smallPca, pcaNames
from sklearn.model_selection import KFold # import KFold
import numpy as np
import sklearn
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier

X = smallPca[0:10, 0:2]
y = smallPca[0:10, -1].squeeze()
attributeNames = pcaNames[0:-1]
N, M = X.shape



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)



kfold = KFold(5, True, 1)
for train, test in kfold.split(X):
	print('train: %s, test: %s' % (X[train], X[test]))


# plt.scatter(y_test, predictions)
# plt.xlabel("True Values")
# plt.ylabel("Predictions")
# plt.show()
#
# print("Score:", model.score(X_test, y_test))


# feature extraction
model = ExtraTreesClassifier()
model.fit(X, y)
print(pcaNames)
print(model.feature_importances_)


