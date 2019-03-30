from sklearn.linear_model import Ridge
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import torch
from tools import train_neural_net


class RegressionModel:
    model = Ridge(alpha=0.6)
    polynomial_features = PolynomialFeatures(degree=1)

    def adjust_to_k(self, k):
        alpha = k / 10
        self.model = Ridge(alpha=alpha)

    def fit(self, x, y):
        x_polynomial = self.polynomial_features.fit_transform(x)
        self.model.fit(x_polynomial, y)

    def predict(self, x):
        x_polynomial = self.polynomial_features.fit_transform(x)
        return self.model.predict(x_polynomial)

    @staticmethod
    def get_name():
        return "regression"


class BaseLineModel:
    y = np.array([])

    def adjust_to_k(self, k):
        pass

    def fit(self, x, y):
        self.y = y.mean()

    def predict(self, x):
        return self.y

    @staticmethod
    def get_name():
        return "baseline"


class ANN:
    y = np.array([])
    n_replicates = 1  # number of networks trained in each k-fold
    max_iter = 5000
    n_hidden_units = 1

    def __init__(self):
        self.loss_fn = torch.nn.MSELoss()  # notice how this is now a mean-squared-error loss
        self.model = lambda *args: None
        self.net = lambda *args: None

    def set_model(self, M):
        self.model = lambda: torch.nn.Sequential(
            torch.nn.Linear(M, self.n_hidden_units),  # M features to n_hidden_units
            torch.nn.Tanh(),  # 1st transfer function,
            torch.nn.Linear(self.n_hidden_units, 1),  # n_hidden_units to 1 output neuron
            # no final tranfer function, i.e. "linear output"
        )

    def adjust_to_k(self, k):
        self.n_hidden_units = k

    def fit(self, x, y):
        n, m = x.shape
        self.set_model(m)
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

    @staticmethod
    def get_name():
        return "artificial neural network"


baseLineModel = BaseLineModel()
regressionModel = RegressionModel()
ann = ANN()
