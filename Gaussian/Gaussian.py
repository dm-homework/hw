
import numpy as np
from sklearn.mixture import GaussianMixture
from itertools import cycle
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.datasets.samples_generator import make_blobs
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
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

gmm = GaussianMixture(n_components=6)
labels=gmm.fit_predict(X)
labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)
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
for k, col in zip(range(n_clusters_), colors):
    my_members = labels == k
    X = PCA(n_components=2).fit_transform(X)
    plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

