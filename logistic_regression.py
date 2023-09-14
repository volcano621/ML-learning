import numpy as np
import pandas as pd


def model(x, weights):
    z = np.dot(x, weights)
    return sigmoid(z)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def logistic_regression(x, y, alpha, num_iterations):
    weights = np.ones((x.shape[1], 1))
    for i in range(num_iterations):
        h = model(x, weights)
        error = h - y.reshape(-1, 1)
        gradient = np.dot(x.T, error) / x.shape[0]
        weights -= alpha * gradient
    return weights


def predict(x,  weights):
    y_hat = model(x, weights)
    y_hard = (y_hat > 0.5) * 1
    return y_hard
