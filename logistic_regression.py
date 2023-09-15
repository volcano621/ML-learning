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


def compute_cost_and_gradient(x, y, weights):
    y = y.reshape(-1, 1)
    m = x.shape[0]
    h = model(x, weights)
    cost = (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    gradient = np.dot(x.T, (h - y)) / m
    return cost, gradient


def adam(x, y, alpha, num_iterations, beta1, beta2, epsilon):
    m, n = x.shape
    weights = m_t = v_t = np.zeros((n, 1))
    t = 0
    for i in range(num_iterations):
        t += 1
        cost, gradient = compute_cost_and_gradient(x, y, weights)
        m_t = beta1 * m_t + (1 - beta1) * gradient
        v_t = beta2 * v_t + (1 - beta2) * np.power(gradient, 2)
        m_t_hat = m_t / (1 - np.power(beta1, t))
        v_t_hat = v_t / (1 - np.power(beta2, t))
        weights -= alpha * m_t_hat / (np.sqrt(v_t_hat) + epsilon)
    return weights


def predict(x,  weights):
    y_hat = model(x, weights)
    y_hard = (y_hat > 0.5) * 1
    return y_hard
