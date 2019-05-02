# exercise 11.1.1
from matplotlib.pyplot import figure, show
import numpy as np
from scipy.io import loadmat
from toolbox_02450 import clusterplot
from sklearn.mixture import GaussianMixture
from project_3.pca import X, y
from project_3.data import get_x_and_y

X,y = get_x_and_y('chd', x_columns=['age', 'tobacco'])

N, M = X.shape
# C = len(classNames)
# Number of clusters
K = 9
cov_type = 'full'  # e.g. 'full' or 'diag'

# define the initialization procedure (initial value of means)
initialization_method = 'random'  # 'random' or 'kmeans'
reps = 20
# number of fits with different initalizations, best result will be kept
# Fit Gaussian mixture model
gmm = GaussianMixture(n_components=K, covariance_type=cov_type, n_init=reps,
                      tol=1e-6, reg_covar=1e-6, init_params=initialization_method).fit(X)
cls = gmm.predict(X)
# extract cluster labels
cds = gmm.means_
# extract cluster centroids (means of gaussians)
covs = gmm.covariances_
# extract cluster shapes (covariances of gaussians)
if cov_type.lower() == 'diag':
    new_covs = np.zeros([K, M, M])

    count = 0
    for elem in covs:
        temp_m = np.zeros([M, M])
        new_covs[count] = np.diag(elem)
        count += 1

    covs = new_covs

# Plot results:
figure(figsize=(14,9))
clusterplot(X, clusterid=cls, centroids=cds, y=y, covars=covs)
show()
print('Ran Exercise 11.1.1')