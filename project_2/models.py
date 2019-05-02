from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import torch
from tools import train_neural_net
from collections import Counter


class RegressionModel:
    model = Ridge(alpha=0.6)
    polynomial_features = PolynomialFeatures(degree=1)
    k_data = {}

    def adjust_to_k(self, k):
        alpha = k / 10 + 0.1
        self.k_data[k] = alpha
        self.model = Ridge(alpha=alpha)

    def fit(self, x, y):
        x_polynomial = self.polynomial_features.fit_transform(x)
        self.model.fit(x_polynomial, y)

    def predict(self, x):
        x_polynomial = self.polynomial_features.fit_transform(x)
        return self.model.predict(x_polynomial)

    def get_name(self):
        return "regression"


class LogisticRegressionModel:
    model = LogisticRegression(penalty='l2', C=10, solver='lbfgs')
    k_data = {}

    def adjust_to_k(self, k):
        alpha = k / 10 + 0.1
        self.k_data[k] = alpha
        self.model = LogisticRegression(penalty='l2', C=1/alpha, solver='lbfgs')

    def fit(self, x, y):
        self.model.fit(x, y.squeeze())

    def predict(self, x):
        return self.model.predict(x)

    def get_name(self):
        return "logistic regression"


class KNN:
    model = KNeighborsClassifier(n_neighbors=1)
    k_data = {}

    def adjust_to_k(self, k):
        neighbours_number = k + 1
        self.k_data[k] = neighbours_number
        self.model = KNeighborsClassifier(n_neighbors=neighbours_number)

    def fit(self, x, y):
        self.model.fit(x, y.squeeze())

    def predict(self, x):
        return self.model.predict(x)

    def get_name(self):
        return "knn"


class BaseClassification:
    y = np.array([])
    k_data = {}

    def adjust_to_k(self, k):
        self.k_data[k] = 0

    def fit(self, x, y):
        (most_common, other) = Counter(y[:, 0]).most_common(1)[0]
        self.most_common = most_common
        self.y = [most_common for i in range(len(y))]

    def predict(self, x):
        return [self.most_common for i in range(len(x))]

    def get_name(self):
        return "baseline classification"



class BaseLineModel:
    y = np.array([])
    k_data = {}


    def adjust_to_k(self, k):
        self.k_data[k] = 0
        pass

    def fit(self, x, y):
        self.y = y.mean()

    def predict(self, x):
        return self.y

    def get_name(self):
        return "baseline"


class ANN:
    y = np.array([])
    n_replicates = 3  # number of networks trained in each k-fold
    max_iter = 1000
    n_hidden_units = 3
    k_data = {}

    def __init__(self, m):
        self.M = m
        self.loss_fn = torch.nn.MSELoss()  # notice how this is now a mean-squared-error loss
        self.set_model()
        self.net = lambda *args: None

    def set_model(self):
        self.model = lambda: torch.nn.Sequential(
            torch.nn.Linear(self.M, self.n_hidden_units),  # M features to n_hidden_units
            torch.nn.Tanh(),  # 1st transfer function,
            torch.nn.Linear(self.n_hidden_units, 1),  # n_hidden_units to 1 output neuron
            # no final tranfer function, i.e. "linear output"
        )

    def adjust_to_k(self, k):
        n_hidden_units = k*2 + 1
        self.k_data[k] = n_hidden_units
        self.n_hidden_units = n_hidden_units
        self.set_model()


    def fit(self, x, y):
        torch_x = torch.tensor(x, dtype=torch.float)
        torch_y = torch.tensor(y, dtype=torch.float)
        self.y = y.mean()
        net, final_loss, learning_curve = train_neural_net(self.model,
                                                           self.loss_fn,
                                                           X=torch_x,
                                                           y=torch_y,
                                                           n_replicates=self.n_replicates,
                                                           max_iter=self.max_iter)
        self.net = net

    def predict(self, x):
        torch_x = torch.tensor(x, dtype=torch.float)
        return self.net(torch_x)

    def get_name(self):
        return "artificial neural network"
