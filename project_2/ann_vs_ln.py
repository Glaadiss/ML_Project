from dataHeartDisaese import x, y
from models import RegressionModel, BaseLineModel, ANN
from sklearn import model_selection

from CrossValidation import CrossValidation

test_proportion = 0.75
x, x_outer, y, y_outer = model_selection.train_test_split(x, y, test_size=test_proportion)
N, M = x.shape

models = lambda: [RegressionModel(), BaseLineModel(), ANN(M)]

outer_cv = CrossValidation(models(), x_outer, y_outer, K=10)
outer_cv.applyInnerValidation(x, y)

outer_cv.test()

outer_cv.print_result_with_k()

outer_cv.show_errors()

outer_cv.compare_all_classifiers()








