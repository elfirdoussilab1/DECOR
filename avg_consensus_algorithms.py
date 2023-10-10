# We implement in this file some algorithms used in Decentralized ML

import numpy as np


def consensus_error(X):
    """
    Computes the quantity 1/n \sum_{i = 1}^{n} \| X[:,i] - x_0 \|_2^2
    Which is the mean of squared distances of each vector to the mean

    Args:
        X (numpy array): matrix containing vectors in its columns
    """
    # computing the mean of columns
    x_0 = np.mean(X, axis=1)
    return np.sum((X.T - x_0) ** 2) / X.shape[1]

def standard_algo(X, topology, num_iter=100):
    # X.shape = (num_dim, num_nodes)

    X_iter = np.copy(X) 

    errors = [consensus_error(X)]
    for i in range(num_iter):
        W = topology(i)
        X_iter = X_iter.dot(W)
        errors.append(consensus_error(X_iter))
    return X_iter, errors

def finite_time_consensus(X, topology, power=1, gamma = 1, num_iter=100):
    # Not sure if it's the exact algo described in the paper (d_ij is not correct because they use a matrix)
    # https://arxiv.org/pdf/2111.02949.pdf
    # X.shape = (num_dim, num_nodes)

    num_dim, num_nodes = X.shape
    X_iter = np.copy(X)

    errors = [consensus_error(X)]
    for it in range(num_iter):
        W = topology(it)
        Delta = np.zeros_like(X_iter)
        for i in range(num_nodes):
            for j in range(num_nodes):
                d_ij = X_iter[:, j] - X_iter[:, i]
                Delta[:, i] += W[i, j] ** power * np.sign(d_ij) * (np.abs(d_ij) ** power)
        X_iter = X_iter + gamma * Delta
        errors.append(consensus_error(X_iter))
    return X_iter, errors



def push_sum(X_init, topology, num_iter=100):
    # X.shape = (num_dim, num_nodes)
    #
    # works with column-stochastic matrices

    X = np.copy(X_init)
    Y = np.ones_like(X)
    errors = [consensus_error(X / Y)]

    for i in range(num_iter):
        W = topology(i)
        X = X.dot(W)
        Y = Y.dot(W)

        errors.append(consensus_error(X / Y))

    return X, errors
