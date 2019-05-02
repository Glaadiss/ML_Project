from sklearn import model_selection
import numpy as np
from sklearn.metrics import mean_squared_error
import torch
from scipy import stats
from matplotlib.pyplot import figure, boxplot, xlabel, ylabel, show, plot, legend
import itertools

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
        self.inner_cv = False
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

    def get_best_model(self):
        items = self.errors.items()
        return sorted(items, key=lambda kv: kv[1]["test"].mean())[0]

    def tune_parameters(self):
        for model in self.models:
            model.adjust_to_k(self.k)
        for model in self.inner_cross_validation.models:
            model.adjust_to_k(self.k)

    def print_result_with_k(self):
        names = [model.get_name() for model in self.models]
        print("outerFold  {0}     {1}      {2}".format(names[0], names[1], names[2]))
        print("              param err     param err    param err")
        params = [{} for k in range(self.K)]
        for j, model in enumerate(self.models):
            for i, err in enumerate(self.errors[model.get_name()]["test"]):
                params[i][j] = "{0}  {1}".format(round(model.k_data[i], 2), round(err, 4))

        for i, param in enumerate(params):
            print("{0}           {1}   {2}  {3} ".format(i, param[0], param[1], param[2]))

    def test(self):
        cv = model_selection.KFold(n_splits=self.K, shuffle=True)
        for train_index, test_index in cv.split(self.x):
            if self.inner_cv:
                self.inner_cross_validation = CrossValidation(self.models, self.inner_x, self.inner_y, self.K)
                self.tune_parameters()
                self.inner_cross_validation.test()
                self.assign_real_values_for_outer()
            else:
                self.assign_real_values(train_index, test_index)
                self.fit_models()

            self.calculate_errors()
            self.k = self.k + 1

    def assign_real_values_for_outer(self):
        self.real_values["x_train"][self.k] = self.x
        self.real_values["x_test"][self.k] = self.x
        self.real_values["y_train"][self.k] = self.y.reshape(-1, 1)
        self.real_values["y_test"][self.k] = self.y.reshape(-1, 1)

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

    def compare_classifiers(self, name_a, name_b):
        error_a = self.errors[name_a]["test"]
        error_b = self.errors[name_b]["test"]
        z = (error_a - error_b)
        zb = z.mean()
        nu = self.K - 1
        sig = (z - zb).std() / np.sqrt(self.K - 1)
        alpha = 0.05
        zL = zb + sig * stats.t.ppf(alpha / 2, nu);
        zH = zb + sig * stats.t.ppf(1 - alpha / 2, nu);
        print("")
        print("{0} vs {1}".format(name_a, name_b))
        print("p: {0}".format(stats.ttest_rel(error_a, error_b)))
        print("zl: {0}  zh:{1}".format(zL, zH))
        if zL <= 0 and zH >= 0:
            print('Classifiers are not significantly different')
        else:
            print('Classifiers are significantly different.')
        print("")

    def compare_all_classifiers(self):
        names = [model.get_name() for model in self.models]
        combs = list(itertools.combinations(names, 2))
        for (a, b) in combs:
            self.compare_classifiers(a, b)


    def show_errors(self):
        x_range = list(range(1, self.K + 1))
        names = [model.get_name() for model in self.models]
        errors = [self.errors[name]["test"] for name in names]
        f = figure()
        for error in errors:
            plot(x_range, error)
        # plot(lambdaRange, y.mean())
        xlabel('K')
        ylabel('error')
        legend(names)
        show()

    def applyInnerValidation(self, x, y):
        self.inner_cv = True
        self.inner_x = x
        self.inner_y = y
