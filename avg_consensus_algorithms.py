import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib
import networkx

def consensus_error(X):
    x_0 = np.mean(X, axis=1)
    return np.sum((X.T - x_0) ** 2) / X.shape[1]

def standart_algo(X, topology, num_iter=100):
    # X.shape = (num_dim, num_nodes)

    X_iter = np.copy(X)

    errors = [consensus_error(X)]
    for i in range(0, num_iter):
        W = topology(i)
        X_iter = X_iter.dot(W)
        errors += [consensus_error(X_iter)]
    return X_iter, errors

def finate_time_consensus(X, topology, power=1, num_iter=100):
    # https://arxiv.org/pdf/2111.02949.pdf
    # X.shape = (num_dim, num_nodes)

    num_dim, num_nodes = X.shape
    X_iter = np.copy(X)

    errors = [consensus_error(X)]
    for it in range(0, num_iter):
        W = topology(it)
        Delta = np.zeros_like(X_iter)
        for i in range(0, num_nodes):
            for j in range(0, num_nodes):
                d_ij = X_iter[:, j] - X_iter[:, i]
                Delta[:, i] += W[i, j] ** power * np.sign(d_ij) * (np.abs(d_ij) ** power)
        X_iter = X_iter + Delta
        errors += [consensus_error(X_iter)]
    return X_iter, errors



def push_sum(X_init, topology, num_iter=100):
    # X.shape = (num_dim, num_nodes)
    #
    # works with column-stochastic matrices

    X = np.copy(X_init)
    Y = np.ones_like(X)
    errors = [consensus_error(X / Y)]

    for i in range(0, num_iter):
        W = topology(i)
        X = X.dot(W)
        Y = Y.dot(W)

        errors += [consensus_error(X / Y)]

    return X, errors
