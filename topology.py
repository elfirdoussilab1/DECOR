# In this file, we implement some Mixing Matrices that we can use in our simulations

import numpy as np
import matplotlib.pyplot as plt
import networkx


def create_ring(n_nodes):
    """
    This function creates a ring, i.e a cycle that connectes all nodes
    Args:
        n_nodes (int): number of nodes
    Return:
        W (numpy array): matrix of weights
    """
    assert (n_nodes % 2) == 0
    assert n_nodes >= 3

    # create W1 first
    W = np.zeros(shape=(n_nodes, n_nodes))
    value = 1./3 if n_nodes >= 3 else 1./2
    np.fill_diagonal(W, value)
    np.fill_diagonal(W[1:], value, wrap=False)
    np.fill_diagonal(W[:, 1:], value, wrap=False)
    W[0, n_nodes - 1] = value
    W[n_nodes - 1, 0] = value
    return W

def create_reversed_ring(n_nodes):
    """
    Creates another ring, but to observe it we need to adjust the placements of the nodes
    Args:
        n_nodes (int): number of nodes
    Return:
        W (numpy array): matrix of weights
    """
    # Divide nodes into two groups
    group_1 = np.arange(0, n_nodes // 2)
    group_2 = np.arange( n_nodes // 2, n_nodes)
    
    # initialize the matrix
    W = np.zeros(shape=(n_nodes, n_nodes))
    value = 1./3 if n_nodes >= 3 else 1./2
    np.fill_diagonal(W, value)

    for i in range(0, n_nodes // 2):
        W[group_1[i], group_2[n_nodes // 2 - i - 1]] = value
        W[group_2[n_nodes // 2 - i - 1], group_1[i]] = value
        if i > 0:
            W[group_1[i], group_2[n_nodes // 2 - i]] = value
            W[group_2[n_nodes // 2 - i], group_1[i]] = value
    W[group_1[0], group_2[0]] = value
    W[group_2[0], group_1[0]] = value

    return W

def avg_two_rings(n_nodes):
    """
    Averages two different ring matrices
    Args:
        n_nodes (int): number of nodes
    Return:
        W (numpy array): matrix of weights
    """
    W1 = create_ring(n_nodes)
    W2 = create_reversed_ring(n_nodes)
    return (W1 + W2) / 2

def centralized(n_nodes):
    """
    Creates a matrix of fully connected nodes
    Args:
        n_nodes (int): number of nodes
    Return:
        W (numpy array): matrix of weights
    """
    W = np.ones((n_nodes, n_nodes), dtype=np.float64) / n_nodes
    return W

def grid(n_nodes):
    """
    Creates a grid
    Args:
        n_nodes (int): number of nodes
    Return:
        W (numpy array): matrix of weights
    """
    assert int(np.sqrt(n_nodes)) ** 2 == n_nodes
    G = networkx.generators.lattice.grid_2d_graph(int(np.sqrt(n_nodes)),
                                        int(np.sqrt(n_nodes)), periodic=True)
    W = networkx.adjacency_matrix(G).toarray()
    for i in range(0, W.shape[0]):
        W[i,i] = 1
    W = W/5
    return W

# Mapping names to functions
topology_names = {'ring': create_ring, 'centralized': centralized, 'avg_two_ring': avg_two_rings, 'grid': grid}

def create_mixing_matrix(topology, n_nodes):
    assert topology in topology_names.keys()
    W = topology_names[topology](n_nodes)
    return W


# Interface of all Mixing Matrices
class MixingMatrix:
    def __call__(self, current_iter):
        pass

class FixedMixingMatrix(MixingMatrix):
    def __init__(self, topology_name, n_nodes):
        self.W = create_mixing_matrix(topology_name, n_nodes)
    def __call__(self, current_iter):
        return self.W

class MixingMatrixSetP(MixingMatrix):
    def __init__(self, n_nodes, p):
        # we take eigenvectors from the ring topology and replace eigenvalues with
        # the [1, p, .... , p]
        W = create_mixing_matrix("ring", n_nodes)
        eigvals = np.linalg.eig(W)[0]
        eigvect = np.real(np.linalg.eig(W)[1])
        V = eigvect[:, np.argsort(eigvals)]
        d = - np.ones_like(eigvals) * p
        d[-1] = 1
        d[-2] =  - p
        d[:-1] /= np.arange(n_nodes - 1) + 1
        self.W = np.real(V.dot(np.diag(d)).dot(np.linalg.inv(V)))
        # self.W = (self.W + self.W.T) * 0.5

    def __call__(self, current_iter):
        return self.W

class MixingMatrixInterpolateCentr(MixingMatrix):
    def __init__(self, n_nodes, coef):
        # interpolated between ring and fully-connected graph
        W_ring = create_mixing_matrix("ring", n_nodes)
        W_full = np.ones_like(W_ring) / n_nodes
        self.W = coef * W_ring + (1 - coef) * W_full

    def __call__(self, current_iter):
        return self.W

class MixingMatrixInterpolateDisc(MixingMatrix):
    def __init__(self, n_nodes, coef):
        # interpolated between ring and disconnected graph
        W_ring = create_mixing_matrix("ring", n_nodes)
        W_disc = np.eye(n_nodes)
        self.W = coef * W_ring + (1 - coef) * W_disc

    def __call__(self, current_iter):
        return self.W


class RabdomMixingMatrix(MixingMatrix):
    def __init__(self, n_nodes):
        # random coefficients for mixing matrix
        # w_{ij} needs to be between 0 and 1, so I choose every coefficient uniformly at random,
        # then symmetrize it and make it stochastic
        is_correct = False
        while not is_correct:
            V = np.random.uniform(size=(n_nodes, n_nodes))
            V = (V + V.T) / n_nodes
            for i in range(0, n_nodes - 1):
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
    def __init__(self, n_nodes, p=0.5):
        self.W1 = create_ring(n_nodes)
        self.W2 = create_reversed_ring(n_nodes)
        self.p = p

    def __call__(self, current_iter):
        idx = np.random.binomial(1, self.p, 1)[0]
        return self.W1 if idx == 0 else self.W2

class AlternatingTwoRings(MixingMatrix):
    def __init__(self, n_nodes):
        self.W1 = create_ring(n_nodes)
        self.W2 = create_reversed_ring(n_nodes)

    def __call__(self, current_iter):
        return self.W1 if current_iter % 2 == 0 else self.W2


class RandomEdge(MixingMatrix):
    def __init__(self, n_nodes):
        self.n_nodes = n_nodes

    def __call__(self, current_iter):
        edge = (0, 0)
        while edge[0] == edge[1]:
            edge = np.random.choice(self.n_nodes), np.random.choice(self.n_nodes)
        W = np.eye(self.n_nodes)
        W[edge[0], edge[0]] = 0.5
        W[edge[1], edge[1]] = 0.5
        W[edge[0], edge[1]] = 0.5
        W[edge[1], edge[0]] = 0.5

        return W



class DirectedRingRowStoch(MixingMatrix):
    '''
    Directed ring with the self-loops of 1/2
    '''
    def __init__(self, n_nodes, is_doubly_stoch=True):
        self.n_nodes = n_nodes

        assert n_nodes >= 2
        W = np.zeros(shape=(n_nodes, n_nodes))
        value = 1./2
        np.fill_diagonal(W, value)
        np.fill_diagonal(W[1:], value, wrap=False)
        W[0, n_nodes - 1] = value

        if not is_doubly_stoch:
            # need to change one of the nodes to have different self-weight
            W[0, 0] = 2./3
            W[0, n_nodes - 1] = 1./3

        self.W = W

    def __call__(self, current_iter):
        return self.W
    # TODO: check if it is row-stochastic






