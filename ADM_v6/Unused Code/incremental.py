from __future__ import print_function

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans, MiniBatchKMeans

import logging
from optparse import OptionParser
import sys
from time import time
from fetchData_Inc import dataSet
import numpy as np


# Display progress logs on stdout

vectorizer = TfidfVectorizer(max_df=0.5,
                             min_df=1, stop_words='english')

#X is a sparce TFIDF matrix
X = vectorizer.fit_transform(dataSet)

print ("Printing Vectorized Matrix (X):\n",X)

print("n_samples: %d, n_features: %d" % X.shape)
print()


# if opts.minibatch:
#     km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
#                          init_size=1000, batch_size=1000, verbose=opts.verbose)
# else:
#     km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,
#                 verbose=opts.verbose)
true_k=4
km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
t0 = time()
km.fit(X)
print("done in %0.3fs" % (time() - t0))
print()

# print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
# print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
# print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
# print("Adjusted Rand-Index: %.3f"
#       % metrics.adjusted_rand_score(labels, km.labels_))
# print("Silhouette Coefficient: %0.3f"
#       % metrics.silhouette_score(X, km.labels_, sample_size=1000))




order_centroids = km.cluster_centers_.argsort()[:, ::-1]

terms = vectorizer.get_feature_names()
for i in range(true_k):#true_k=4
    print("Cluster %d:" % i, end='')
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind], end='')
    print()

print ("Cluster Centres: ",km.cluster_centers_)
print("Cluster Results: ",km.labels_)