import numpy as np
from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, legend
from scipy.linalg import svd
from project_3.data import x2 as X, y2

classNames = ['not ill', 'ill']
C = len(classNames)
N, M = X.shape
Y = X - np.ones((N, 1)) * X.mean(0)
U, S, Vh = svd(Y, full_matrices=False)
V = Vh.T
Z = Y @ V

X = Z[:,[0, 1]]
y = y2
# Indices of the principal components to be plotted
# i = 0
# j = 1
#
# # Plot PCA of the data
# f = figure()
# title('congenital heart defect')
# # Z = array(Z)
# for c in range(C):
#     # select indices belonging to class c:
#     class_mask = y == c
#     plot(Z[class_mask, i], Z[class_mask, j], 'o', alpha=.5)
# legend(classNames)
# xlabel('PC{0}'.format(i + 1))
# ylabel('PC{0}'.format(j + 1))
#
# # Output result to screen
# show()


print('Ran Exercise 2.1.4')
