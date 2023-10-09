import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib
import networkx


def create_two_rings(n_cores):
    assert (n_cores % 2) == 0
    assert n_cores >= 3
    # create W1 first
    W = np.zeros(shape=(n_cores, n_cores))
    value = 1./3 if n_cores >= 3 else 1./2
    np.fill_diagonal(W, value)
    np.fill_diagonal(W[1:], value, wrap=False)
    np.fill_diagonal(W[:, 1:], value, wrap=False)
    W[0, n_cores - 1] = value
    W[n_cores - 1, 0] = value
    
    W1 = W
    
    group_1 = np.arange(0, n_cores // 2)
    group_2 = np.arange( n_cores // 2, n_cores)
    # print(group_1, group_2)
    W = np.zeros(shape=(n_cores, n_cores))
    value = 1./3 if n_cores >= 3 else 1./2
    np.fill_diagonal(W, value)
    for i in range(0, n_cores // 2):
        W[group_1[i], group_2[n_cores // 2 - i - 1]] = value
        W[group_2[n_cores // 2 - i - 1], group_1[i]] = value
        if i > 0:
            W[group_1[i], group_2[n_cores // 2 - i]] = value
            W[group_2[n_cores // 2 - i], group_1[i]] = value
    W[group_1[0]][group_2[0]] = value
    W[group_2[0]][group_1[0]] = value
    W2 = W
    return W1, W2

def create_mixing_matrix(topology, n_cores):
    assert topology in ['ring', 'centralized', 'grid', 'avg_two_ring']
    if topology == 'avg_two_ring':
        W1, W2 = create_two_rings(n_cores)
        return (W1 + W2) * 0.5
    elif topology == 'ring':
        W = np.zeros(shape=(n_cores, n_cores))
        value = 1./3 if n_cores >= 3 else 1./2
        np.fill_diagonal(W, value)
        np.fill_diagonal(W[1:], value, wrap=False)
        np.fill_diagonal(W[:, 1:], value, wrap=False)
        W[0, n_cores - 1] = value
        W[n_cores - 1, 0] = value
        return W
    elif topology == 'centralized':
        W = np.ones((n_cores, n_cores), dtype=np.float64) / n_cores
        return W
    else:
        assert int(np.sqrt(n_cores)) ** 2 == n_cores
        G = networkx.generators.lattice.grid_2d_graph(int(np.sqrt(n_cores)),
                                            int(np.sqrt(n_cores)), periodic=True)
        # print(G)
        W = networkx.adjacency_matrix(G).toarray()
        for i in range(0, W.shape[0]):
            W[i][i] = 1
        W = W/5
        return W



class MixingMatrix:
    def __call__(self, current_iter):
        pass

class FixedMixingMatrix(MixingMatrix):
    def __init__(self, topology_name, n_cores):
        self.W = create_mixing_matrix(topology_name, n_cores)
    def __call__(self, current_iter):
        return self.W

class MixingMatrixSetP(MixingMatrix):
    def __init__(self, n_cores, p):
        # we take eigenvectors from the ring topology and replace eigenvalues with
        # the [1, p, .... , p]
        W = create_mixing_matrix("ring", n_cores)
        eigvals = np.linalg.eig(W)[0]
        eigvect = np.real(np.linalg.eig(W)[1])
        V = eigvect[:, np.argsort(eigvals)]
        d = - np.ones_like(eigvals) * p
        d[-1] = 1
        d[-2] =  - p
        d[:-1] /= np.arange(n_cores - 1) + 1
        self.W = np.real(V.dot(np.diag(d)).dot(np.linalg.inv(V)))
        # self.W = (self.W + self.W.T) * 0.5

    def __call__(self, current_iter):
        return self.W

class MixingMatrixInterpolateCentr(MixingMatrix):
    def __init__(self, n_cores, coef):
        # interpolated between ring and fully-connected graph
        W_ring = create_mixing_matrix("ring", n_cores)
        W_full = np.ones_like(W_ring) / n_cores
        self.W = coef * W_ring + (1 - coef) * W_full

    def __call__(self, current_iter):
        return self.W

class MixingMatrixInterpolateDisc(MixingMatrix):
    def __init__(self, n_cores, coef):
        # interpolated between ring and disconnected graph
        W_ring = create_mixing_matrix("ring", n_cores)
        W_disc = np.eye(n_cores)
        self.W = coef * W_ring + (1 - coef) * W_disc

    def __call__(self, current_iter):
        return self.W


class RabdomMixingMatrix(MixingMatrix):
    def __init__(self, n_cores):
        # random coefficients for mixing matrix
        # w_{ij} needs to be between 0 and 1, so I choose every coefficient uniformly at random,
        # then symmetrize it and make it stochastic
        is_correct = False
        while not is_correct:
            V = np.random.uniform(size=(n_cores, n_cores))
            V = (V + V.T) / n_cores
            for i in range(0, n_cores - 1):
                V[i][-1] = 1 - np.sum(V[i][:-1])
                V[-1][i] = V[i][-1]
            V[-1][-1] = 1 - np.sum(V[-1][:-1])
            # Check that all the eigenvalues are smaller than 1.
            if np.max(np.abs(np.linalg.eigvals(V))) <= 1. + 1e-15:
                is_correct = True
        self.W = V


    def __call__(self, current_iter):
        return self.W


class RandomizedTwoRings(MixingMatrix):
    def __init__(self, n_cores, p=0.5):
        self.W1, self.W2 = create_two_rings(n_cores)
        self.p = p

    def __call__(self, current_iter):
        idx = np.random.binomial(1, self.p, 1)[0]
        return self.W1 if idx == 0 else self.W2

class AlternatingTwoRings(MixingMatrix):
    def __init__(self, n_cores):
        self.W1, self.W2 = create_two_rings(n_cores)

    def __call__(self, current_iter):
        return self.W1 if current_iter % 2 == 0 else self.W2


class RandomEdge(MixingMatrix):
    def __init__(self, n_cores):
        self.n_cores = n_cores

    def __call__(self, current_iter):
        edge = (0, 0)
        while edge[0] == edge[1]:
            edge = np.random.choice(self.n_cores), np.random.choice(self.n_cores)
        W = np.eye(self.n_cores)
        W[edge[0], edge[0]] = 0.5
        W[edge[1], edge[1]] = 0.5
        W[edge[0], edge[1]] = 0.5
        W[edge[1], edge[0]] = 0.5

        return W



class DirectedRingRowStoch(MixingMatrix):
    '''
    Directed ring with the self-loops of 1/2
    '''
    def __init__(self, n_cores, is_doubly_stoch=True):
        self.n_cores = n_cores

        assert n_cores >= 2
        W = np.zeros(shape=(n_cores, n_cores))
        value = 1./2
        np.fill_diagonal(W, value)
        np.fill_diagonal(W[1:], value, wrap=False)
        W[0, n_cores - 1] = value

        if not is_doubly_stoch:
            # need to change one of the nodes to have different self-weight
            W[0, 0] = 2./3
            W[0, n_cores - 1] = 1./3

        self.W = W

    def __call__(self, current_iter):
        return self.W
    # TODO: check if it is row-stochastic






