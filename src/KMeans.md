# K-Means Algorithm

K-Means is a popular unsupervised learning algorithm used for clustering data into distinct groups based on their features. It is widely used in various fields such as customer segmentation, image compression, and pattern recognition.

## Table of Contents
- [Introduction](#introduction)
- [Algorithm Steps](#algorithm-steps)
- [Mathematical Equations](#mathematical-equations)
- [Code Implementation](#code-implementation)

## Introduction

K-Means clustering partitions the dataset into $( k )$ clusters, where each data point belongs to the cluster with the nearest mean. The algorithm iteratively updates the centroids (means) of the clusters and assigns data points to the closest centroid until convergence.

## Algorithm Steps

1. **Initialize centroids**: Randomly select $( k )$ data points as the initial centroids.
2. **Assign clusters**: Assign each data point to the nearest centroid based on the Euclidean distance.
3. **Update centroids**: Calculate the mean of all data points assigned to each cluster to update the centroids.
4. **Repeat**: Repeat steps 2 and 3 until the centroids no longer change (convergence) or for a fixed number of iterations.

## Mathematical Equations

1. Distance Calculation

    The Euclidean distance between a data point $( \mathbf{x}_i )$ and a centroid $( \mathbf{\mu}_j )$ is given by:

    $d(\mathbf{x}_i, \mathbf{\mu}_j) = \sqrt{\sum_{m=1}^{n} (x_{im} - \mu_{jm})^2}$

    Where:
    - $( \mathbf{x}_i )$ is the $( i )$-th data point.
    - $( \mathbf{\mu}_j )$ is the $( j )$-th centroid.
    - $( n )$ is the number of features.

2. Centroid Update

    The new centroid $( \mathbf{\mu}_j )$ for a cluster is calculated as the mean of all data points assigned to that cluster:

    $\mathbf{\mu}_j = \frac{1}{|C_j|} \sum_{\mathbf{x}_i \in C_j} \mathbf{x}_i$

    Where:
    - $( C_j )$ is the set of data points assigned to cluster $( j )$.
    - $( |C_j| )$ is the number of data points in cluster $( j )$.

3. Objective Function

    The objective of K-Means is to minimize the within-cluster sum of squares (WCSS), also known as inertia:

    $\text{WCSS} = \sum_{j=1}^{k} \sum_{\mathbf{x}_i \in C_j} \|\mathbf{x}_i - \mathbf{\mu}_j\|^2$

    Where:
    - $( k )$ is the number of clusters.
    - $( \|\mathbf{x}_i - \mathbf{\mu}_j\|^2 )$ is the squared Euclidean distance between a data point and its centroid.

## Code Implementation

Here is an example of implementing K-Means using Python with the help of the `scikit-learn` library:

```python
import numpy as np

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KMeans:
    def __init__(self, K=5, max_iters=100):
        self.K = K
        self.max_iters = max_iters

        # List of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)]

        # The centers (mean vector) for each cluster
        self.centroids = []

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        # Initialising centroids
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        # Optimizing Clusters
        for _ in range(self.max_iters):
            # Assign samples to closest centroids (create clusters)
            self.clusters = self._create_clusters(self.centroids)

            # Calculate new centroids from the clusters
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            if self._is_converged(centroids_old, self.centroids):
                break

        # Classify samples as the index of their clusters
        return self._get_cluster_labels(self.clusters)

    def _get_cluster_labels(self, clusters):
        # Each sample will get the label of the cluster it was assigned to
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx

        return labels

    def _create_clusters(self, centroids):
        # Assign the samples to the closest centroids
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        # Distance of the current sample to each centroid
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx

    def _get_centroids(self, clusters):
        # Assign mean value of clusters to centroids
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, centroids_old, centroids):
        # Distances between old and new centroids, for all centroids
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0