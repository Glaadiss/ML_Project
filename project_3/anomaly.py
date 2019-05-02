# exercise 11.4.1
import numpy as np
from matplotlib.pyplot import (figure, bar, title, show)
from sklearn.neighbors import NearestNeighbors

from project_3.data import get_x_and_y, df
from toolbox_02450 import gausKernelDensity

X, y = get_x_and_y('chd', yNominal=True, x_columns=['age', 'tobacco'])
N, M = X.shape

### Gausian Kernel density estimator
# cross-validate kernel width by leave-one-out-cross-validation
# (efficient implementation in gausKernelDensity function)
# evaluate for range of kernel widths
widths = X.var(axis=0).max() * (2.0 ** np.arange(-10, 3))
logP = np.zeros(np.size(widths))
for i, w in enumerate(widths):
    print('Fold {:2d}, w={:f}'.format(i, w))
    density, log_density = gausKernelDensity(X, w)
    logP[i] = log_density.sum()

val = logP.max()
ind = logP.argmax()

width = widths[ind]
print('Optimal estimated width is: {0}'.format(width))

# evaluate density for estimated width
density, log_density = gausKernelDensity(X, width)

# Sort the densities
i = (density.argsort(axis=0)).ravel()
density = density[i].reshape(-1, )

for j in range(5):
    print(X[i[j]])

# Plot density estimate of outlier score
figure(1)
bar(range(N), density[:N], width=2)
title('Density estimate')
show()
### K-neighbors density estimator
# Neighbor to use:
K = 5

# Find the k nearest neighbors
knn = NearestNeighbors(n_neighbors=K).fit(X)
D, i = knn.kneighbors(X)

density = 1. / (D.sum(axis=1) / K)

# Sort the scores
i = density.argsort()
density = density[i]
for j in range(5):
    print(X[i[j]])
# Plot k-neighbor estimate of outlier score (distances)
figure(3)
bar(range(N), density[:N], width=2)
title('KNN density: Outlier score')

### K-nearest neigbor average relative density
# Compute the average relative density

knn = NearestNeighbors(n_neighbors=K).fit(X)
D, i = knn.kneighbors(X)
density = 1. / (D.sum(axis=1) / K)
avg_rel_density = density / (density[i[:, 1:]].sum(axis=1) / K)

# Sort the avg.rel.densities
i_avg_rel = avg_rel_density.argsort()
avg_rel_density = avg_rel_density[i_avg_rel]
for j in range(5):
    print(X[i_avg_rel[j]])

# Plot k-neighbor estimate of outlier score (distances)
figure(5)
bar(range(N), avg_rel_density[:N], width=2)
title('KNN average relative density: Outlier score')

show()
