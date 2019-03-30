from sklearn import model_selection
import numpy as np
from sklearn.metrics import mean_squared_error
import torch


def k_container(k):
    return [np.empty(k) for el in range(0, 10)]


def calculate_error(predicted_y, real_y):
    if isinstance(predicted_y, torch.Tensor):
        se = (predicted_y - torch.tensor(real_y, dtype=torch.float)) ** 2  # squared error
        return (sum(se).type(torch.float) / len(real_y)).data.numpy()  # mean
    if not isinstance(predicted_y, np.ndarray):
        predicted_y = np.ones(len(real_y)) * predicted_y
    return mean_squared_error(real_y, predicted_y)


class CrossValidation:
    def __init__(self, models, x, y, K):
        self.inner_cross_validation = None
        self.K = K
        self.k = 0
        self.models = models
        self.x = x
        self.y = y
        self.errors = {}
        for model in models:
            self.errors[model.get_name()] = {
                "train": np.empty(K),
                "test": np.empty(K)
            }

        self.real_values = {
            "x_train": k_container(K),
            "x_test": k_container(K),
            "y_train": k_container(K),
            "y_test": k_container(K),
        }

    def apply_inner_cross_validation(self, validation):
        self.inner_cross_validation = validation

    def get_best_model(self):
        items = self.errors.items()
        return sorted(items, key=lambda kv: kv[1]["test"].mean())[0]

    def test(self):
        cv = model_selection.KFold(n_splits=self.K, shuffle=True)
        for train_index, test_index in cv.split(self.x):
            if self.inner_cross_validation:
                self.inner_cross_validation.test()

            self.assign_real_values(train_index, test_index)
            self.fit_models()
            self.calculate_errors()
            self.k = self.k + 1

    def assign_real_values(self, train_index, test_index):
        self.real_values["x_train"][self.k] = self.x[train_index, :]
        self.real_values["x_test"][self.k] = self.x[test_index, :]
        self.real_values["y_train"][self.k] = self.y[train_index].reshape(-1, 1)
        self.real_values["y_test"][self.k] = self.y[test_index].reshape(-1, 1)

    def fit_models(self):
        for model in self.models:
            model.fit(self.real_values["x_train"][self.k],
                      self.real_values["y_train"][self.k])

    def calculate_errors(self):
        predictions = self.predict()
        for model in self.models:
            self.assign_error(model.get_name(), "train", predictions)
            self.assign_error(model.get_name(), "test", predictions)

    def predict(self):
        predictions = {}
        for model in self.models:
            predictions[model.get_name()] = {
                "train": model.predict(self.real_values["x_train"][self.k]),
                "test": model.predict(self.real_values["x_test"][self.k])
            }
        return predictions

    def assign_error(self, model, data_type, predictions):
        prediction = predictions[model][data_type]
        real_value = self.real_values["y_" + data_type][self.k]
        self.errors[model][data_type][self.k] = calculate_error(prediction, real_value)

    def get_mean_errors(self):
        errors = {}
        for model in self.models:
            name = model.get_name()
            errors[name] = round(self.errors[name]["test"].mean(), 5)
        return errors
