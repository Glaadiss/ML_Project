# exercise 7.1.2

from matplotlib.pyplot import figure, plot, xlabel, ylabel, show
import numpy as np
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from data import normalizedX, normalizedY

X = normalizedX
y = [1 if x > 0 else 0 for x in normalizedY]
y = np.array(y)
N = 10

# Maximum number of neighbors
L = 10

CV = model_selection.KFold(N)
errors = np.zeros((N, L))
i = 0
for train_index, test_index in CV.split(X, y):
    print('Crossvalidation fold: {0}/{1}'.format(i + 1, N))

    # extract training and test set for current CV fold
    X_train = X[train_index, :]
    y_train = y[train_index]
    X_test = X[test_index, :]
    y_test = y[test_index]

    # Fit classifier and classify the test points (consider 1 to 40 neighbors)
    for l in range(1, L + 1):
        knclassifier = KNeighborsClassifier(n_neighbors=l);
        knclassifier.fit(X_train, y_train);
        y_est = knclassifier.predict(X_test);
        errors[i, l - 1] = np.sum(y_est[0] != y_test[0])

    i += 1

# Plot the classification error rate
figure()
plot(100 * sum(errors, 0) / N)
xlabel('Number of neighbors')
ylabel('Classification error rate (%)')
show()

print('Ran Exercise 7.1.2')