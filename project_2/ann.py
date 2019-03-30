import torch
from dataHeartDisaese import x as X, y
from sklearn import model_selection
from tools import train_neural_net
import numpy as np

N, M = X.shape
# Parameters for neural network classifier
n_hidden_units = 3      # number of hidden units
n_replicates = 1        # number of networks trained in each k-fold
max_iter = 5000        #

# K-fold crossvalidation
K = 10                   # only three folds to speed up this example
CV = model_selection.KFold(K, shuffle=True)


class ANN:
    y = np.array([])
    n_replicates = 1  # number of networks trained in each k-fold

    def __init__(self, n_hidden_units):
        self.model = lambda: torch.nn.Sequential(
            torch.nn.Linear(M, n_hidden_units),  # M features to n_hidden_units
            torch.nn.Tanh(),  # 1st transfer function,
            torch.nn.Linear(n_hidden_units, 1),  # n_hidden_units to 1 output neuron
            # no final tranfer function, i.e. "linear output"
        )
        self.loss_fn = torch.nn.MSELoss()  # notice how this is now a mean-squared-error loss

    def fit(self, x, y):
        torch_x = torch.tensor(x, dtype=torch.float)
        torch_y = torch.tensor(y, dtype=torch.float)
        self.y = y.mean()
        net, final_loss, learning_curve = train_neural_net(model,
                                                           loss_fn,
                                                           X=torch_x,
                                                           y=torch_y,
                                                           n_replicates=n_replicates,
                                                           max_iter=max_iter)
        self.net = net


    def predict(self, x):
        torch_x = torch.tensor(x, dtype=torch.float)
        return self.net(torch_x)

    @staticmethod
    def get_name():
        return "baseline"

#
#
#
# model = lambda: torch.nn.Sequential(
#     torch.nn.Linear(M, n_hidden_units),  # M features to n_hidden_units
#     torch.nn.Tanh(),  # 1st transfer function,
#     torch.nn.Linear(n_hidden_units, 1),  # n_hidden_units to 1 output neuron
#     # no final tranfer function, i.e. "linear output"
# )
# loss_fn = torch.nn.MSELoss()  # notice how this is now a mean-squared-error loss
#
# print('Training model of type:\n\n{}\n'.format(str(model())))
# errors = []  # make a list for storing generalizaition error in each loop
# for (k, (train_index, test_index)) in enumerate(CV.split(X, y)):
#     print('\nCrossvalidation fold: {0}/{1}'.format(k + 1, K))
#
#     # Extract training and test set for current CV fold, convert to tensors
#     X_train = torch.tensor(X[train_index, :], dtype=torch.float)
#     y_train = torch.tensor(y[train_index], dtype=torch.float)
#     X_test = torch.tensor(X[test_index, :], dtype=torch.float)
#     y_test = torch.tensor(y[test_index], dtype=torch.uint8)
#
#     # Train the net on training data
#     net, final_loss, learning_curve = train_neural_net(model,
#                                                        loss_fn,
#                                                        X=X_train,
#                                                        y=y_train,
#                                                        n_replicates=n_replicates,
#                                                        max_iter=max_iter)
#
#     print('\n\tBest loss: {}\n'.format(final_loss))
#     y_test_est = net(X_test)
#
#     # Determine errors and errors
#     se = (y_test_est.float() - y_test.float()) ** 2  # squared error
#     mse = (sum(se).type(torch.float) / len(y_test)).data.numpy()  # mean
#     errors.append(mse)  # store error rate for current CV fold
#
# print(errors)