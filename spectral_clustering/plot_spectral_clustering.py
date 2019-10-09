"""
=============================================
A demo of the mean-shift clustering algorithm
=============================================

Reference:

Dorin Comaniciu and Peter Meer, "Mean Shift: A robust approach toward
feature space analysis". IEEE Transactions on Pattern Analysis and
Machine Intelligence. 2002. pp. 603-619.

"""
print(__doc__)

import numpy as np
from sklearn.cluster import SpectralClustering
from itertools import cycle
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.datasets.samples_generator import make_blobs
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from time import time
from sklearn.cluster import KMeans

np.random.seed(42)

digits = load_digits()
data = scale(digits.data)

n_samples, n_features = data.shape
X = data

n_digits = len(np.unique(digits.target))
labels_true = digits.target

# #############################################################################
sp = SpectralClustering(n_clusters=n_digits, affinity='nearest_neighbors')
sp.fit(X)
labels = sp.labels_



print("Normalized: %0.3f" % metrics.normalized_mutual_info_score(labels_true, labels))
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))

print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
# # #############################################################################
# Plot result
import matplotlib.pyplot as plt
from itertools import cycle

plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(1000), colors):
    my_members = labels == k
    X = PCA(n_components=2).fit_transform(X)
    plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
  
plt.show()

