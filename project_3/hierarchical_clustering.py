# exercise 10.2.1
from matplotlib.pyplot import figure, show
from scipy.io import loadmat
from toolbox_02450 import clusterplot
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, distance
from project_3.data import origX

# data = 1 - origX.corr()
# corr_condensed = distance.squareform(data) # convert to condensed
# z = linkage(data, method='centroid')
# dendrogram(z, labels=data.columns)
# show()

data = origX
corr_condensed = distance.squareform(data) # convert to condensed
z = linkage(data, method='centroid')
dendrogram(z)
show()


print('Ran Exercise 10.2.1')