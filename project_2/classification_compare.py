from dataHeartDisaese import x2, y2
from models import KNN, LogisticRegressionModel, BaseClassification
from sklearn import model_selection

from CrossValidation import CrossValidation

test_proportion = 0.8
x, x_outer, y, y_outer = model_selection.train_test_split(x2, y2, test_size=test_proportion)
N, M = x.shape

models = lambda: [BaseClassification(), KNN(), LogisticRegressionModel()]

outer_cv = CrossValidation(models(), x_outer, y_outer, K=10)
outer_cv.applyInnerValidation(x, y)

outer_cv.test()

outer_cv.print_result_with_k()

outer_cv.show_errors()

outer_cv.compare_all_classifiers()







