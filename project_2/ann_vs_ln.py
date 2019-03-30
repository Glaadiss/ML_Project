from dataHeartDisaese import x, y
from models import regressionModel, baseLineModel, ann
from sklearn import model_selection

from CrossValidation import CrossValidation

test_proportion = 0.8
x, x_outer, y, y_outer = model_selection.train_test_split(x, y, test_size=test_proportion)
N, M = x.shape

models = [regressionModel, baseLineModel, ann]
outer_cross_validation = CrossValidation(models, x, y, K=3)
inner_cross_validation = CrossValidation(models, x, y, K=10)
inner_cross_validation.test()

# print(inner_cross_validation.get_best_model())


print(inner_cross_validation.get_mean_errors())