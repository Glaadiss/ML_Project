# exercise 6.2.1
from matplotlib.pyplot import figure, plot,  xlabel, ylabel, show, legend, xscale
import numpy as np
from dataHeartDisaese import x as X, y
from sklearn import model_selection
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge


N, M = X.shape

## Crossvalidation
K = 10
CV = model_selection.KFold(n_splits=K,shuffle=True)

# Initialize variables
lambdaRange = np.arange(0.0001, 1, 0.1)
# lambdaRange = np.log([1, 2, 3, 4, 10])
Error_train = np.empty((len(lambdaRange), K))
Error_test = np.empty((len(lambdaRange), K))

k = 0

for train_index, test_index in CV.split(X):
    print('Computing CV fold: {0}/{1}..'.format(k+1,K))

    # extract training and test set for current CV fold
    X_train, y_train = X[train_index, :], y[train_index].reshape(-1, 1)
    X_test, y_test = X[test_index, :], y[test_index].reshape(-1, 1)
    polynomial_features = PolynomialFeatures(degree=1)
    X_poly_train = polynomial_features.fit_transform(X_train)
    X_poly_test = polynomial_features.fit_transform(X_test)

    for i, t in enumerate(lambdaRange):
        model = Ridge(alpha=t)
        model.fit(X_poly_train, y_train)
        y_est_test = model.predict(X_poly_test)
        y_est_train = model.predict(X_poly_train)
        # Evaluate misclassification rate over train/test data (in this CV fold)
        Error_test[i, k] = sum(np.square(y_est_test - y_test)) / float(len(y_est_test))
        Error_train[i, k] = sum(np.square(y_est_train - y_train)) / float(len(y_est_train))

    k += 1

err, bestLambda = sorted(list(zip(Error_test.mean(1), lambdaRange)))[0]
print('Best lambda for model {0}'.format(round(bestLambda, 2)))
f = figure()
plot(lambdaRange, Error_train.mean(1))
plot(lambdaRange, Error_test.mean(1))
# plot(lambdaRange, y.mean())
xlabel('lambda')
ylabel('error'.format(K))
legend(['Error_train', 'Error_test'])
# xscale('log')
show()


