# exercise 6.2.1
from matplotlib.pyplot import figure, plot,  xlabel, ylabel, show, legend
import numpy as np
from data import pcaNames, smallPca, linearRegressionData, normalizedRegressionData
from sklearn import model_selection
from sklearn.linear_model import Ridge
import sklearn.linear_model as lm

X = normalizedRegressionData[:, list(range(len(normalizedRegressionData[0]) - 1))]
y = normalizedRegressionData[:, -1].squeeze()
attributeNames = pcaNames[0:-1]
N, M = X.shape

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(n_splits=K,shuffle=True)

# Initialize variables
lambdaRange = np.arange(0.01, 10, 0.5)
Error_train = np.empty((len(lambdaRange),K))
Error_test = np.empty((len(lambdaRange),K))

k=0
for train_index, test_index in CV.split(X):
    print('Computing CV fold: {0}/{1}..'.format(k+1,K))

    # extract training and test set for current CV fold
    X_train, y_train = X[train_index,:], y[train_index]
    X_test, y_test = X[test_index,:], y[test_index]

    for i, t in enumerate(lambdaRange):
        # Fit decision tree classifier, Gini split criterion, different pruning levels
        # Compute squared error with all features selected (no feature selection)
        # clf = Ridge(alpha=i)
        clf = Ridge(alpha=t)
        # clf = lm.LinearRegression()
        clf.fit(X, y)
        y_est_test = clf.predict(X_test)
        y_est_train = clf.predict(X_train)
        # Evaluate misclassification rate over train/test data (in this CV fold)
        Error_test[i, k]= sum(np.square(y_est_test - y_test)) / float(len(y_est_test))
        Error_train[i, k] = sum(np.square(y_est_train - y_train)) / float(len(y_est_train))

    k += 1

f = figure()
plot(lambdaRange, Error_train.mean(1))
plot(lambdaRange, Error_test.mean(1))
xlabel('lambda')
ylabel('error'.format(K))
legend(['Error_train', 'Error_test'])

show()

