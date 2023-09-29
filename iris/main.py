import pandas as pd
import matplotlib.pyplot as plt
from kmeans import KMeans


if __name__ == "__main__":
    df = pd.read_csv("dataset/Iris.csv")
    df.drop("Id", axis=1, inplace=True)
    x = df.iloc[:, :-1].values  # :-1 means all columns except the last one
    y = df.iloc[:, -1].values  # -1 means the last column

    kmeans = KMeans(3)

    centroids, clusters = kmeans.fit(x, 1000)
    plt.scatter(x[clusters == 0, 0], x[clusters == 0, 1], s=100, c='red', label='Iris-setosa')
    plt.scatter(x[clusters == 1, 0], x[clusters == 1, 1], s=100, c='blue', label='Iris-versicolour')
    plt.scatter(x[clusters == 2, 0], x[clusters == 2, 1], s=100, c='green', label='Iris-virginica')
    plt.scatter(centroids[:, 0], centroids[:, 1], s=100, c='yellow', label='Centroids')
    plt.show()

