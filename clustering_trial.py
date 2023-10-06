from sklearn.cluster import AgglomerativeClustering,HDBSCAN
from sklearn.datasets import make_blobs
import numpy as np
# X = np.array([[1, 2,3], [1, 4,5], [1, 0,3],
#               [4, 2,5], [4, 4,4], [4, 0,10]],dtype=float)
X, y = make_blobs(n_samples=1000000, centers=15, n_features=3,random_state=0)
print(X.shape)
clustering = HDBSCAN(min_cluster_size=5,metric="euclidean",cluster_selection_epsilon=3.55,n_jobs=-1,allow_single_cluster=True,store_centers='centroid').fit(X)
print(clustering.labels_)