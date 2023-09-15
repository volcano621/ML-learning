import numpy as np


class KMeans:
    def __init__(self, K):
        self.K = K
        self.centroids = None

    def initialize_centroids(self, x):
        self.centroids = x[np.random.choice(x.shape[0], self.K, replace=False)]

    def assign_points_centroids(self, X):
        X = np.expand_dims(X, axis=1)  # 增加一个维度，与centroids维度一致
        distance = np.linalg.norm((X - self.centroids), axis=-1)
        points = np.argmin(distance, axis=1)  # argmin返回最小值的索引
        return points

    def compute_mean(self, X, points):
        centroids = np.zeros((self.K, X.shape[1]))
        for i in range(self.K):
            centroid_mean = X[points == i].mean(axis=0)  # points == i: 返回points中等于i的索引, axis=0: 按列求均值
            centroids[i] = centroid_mean
        return centroids

    def fit(self, X, iterations=10):
        self.initialize_centroids(X)
        points = None
        for i in range(iterations):
            points = self.assign_points_centroids(X)
            self.centroids = self.compute_mean(X, points)
        return self.centroids, points
