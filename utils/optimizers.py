from typing import Any
import numpy as np
import os
import time

# This function is the same as standard_algo in avg_consensus_algorithms

def consensus_distance(X, A, B): # ||x-x*||^2
    # X.shape = (num_dim, num_nodes)

    x_star = np.linalg.inv(np.einsum("ikj,ikl->jl", A, A)).dot(np.einsum("ijk,ij->k", A, B))
    num_nodes = X.shape[1]
    dist = [np.linalg.norm(X[:,i] - x_star) ** 2 for i in range(num_nodes)]
    return np.mean(dist)


def optimize_averaging(X, topology, num_iter):
    
    # X.shape = (num_dim, num_nodes)
    def error(X, X_iter):
        x_star = np.mean(X, axis=1)
        num_nodes = X.shape[1]
        dist = [np.linalg.norm(X_iter[:,i] - x_star) ** 2 for i in range(0, num_nodes)]
        return np.mean(dist)

    X_iter = np.copy(X)
    errors = [error(X, X_iter)]
    for i in range(num_iter):
        W = topology(i)
        X_iter = X_iter.dot(W)
        errors.append(error(X, X_iter))
    return errors, X_iter


def optimize_decentralized(X, topology, A, B, gamma, sigma, num_iter=100):
    # X.shape = (num_dim, num_nodes)
    # A.shape = (num_nodes, num_dim, num_dim)
    # B.shape = (num_nodes, num_dim)
    
    num_dim, num_nodes = X.shape
    X_iter = np.copy(X)
    errors = [consensus_distance(X_iter, A, B)]
    for i in range(0, num_iter):
        W = topology(i)
        AXmB = (np.einsum("ijk,ik->ij", A, X_iter.T) - B) # shape (num_nodes, num_dim)
        grad = np.einsum("ijk,ij->ik", A, AXmB) # shape (num_nodes, num_dim)
        noise = np.random.normal(0, np.sqrt(sigma / num_dim), size=X.shape)
        X_iter = X_iter - gamma * (grad.T + noise)
        X_iter = X_iter.dot(W)
        errors.append(consensus_distance(X_iter, A, B))
    return errors, X_iter


def optimize_decentralized_correlated(X, topology, A, B, gamma, sigma, sigma_cor, c_clip, num_gossip=1, num_iter=100, uncorrelate=False):
    # X.shape = (num_dim, num_nodes)
    # A.shape = (num_nodes, num_dim, num_dim)
    # B.shape = (num_nodes, num_dim)

    num_dim, num_nodes = X.shape
    X_iter = np.copy(X)
    errors = [consensus_distance(X_iter, A, B)]
    for i in range(0, num_iter):
        W = topology(i)

        W = np.linalg.matrix_power(W, num_gossip)
        AXmB = (np.einsum("ijk,ik->ij", A, X_iter.T) - B)  # shape (num_nodes, num_dim)
        grad = np.einsum("ijk,ij->ik", A, AXmB)  # shape (num_nodes, num_dim)
        clip_matrix = np.diag(np.minimum(1, c_clip / np.linalg.norm(grad, axis=1)))
        grad = clip_matrix @ grad

        noise = np.random.normal(0, sigma, size=X.shape)
        # initializing and adding correlated noise
        cor_noise = np.zeros_like(X_iter)

        if sigma_cor != 0 and not (W != 0).all():
            for j in range(num_nodes):
                for k in range(j+1, num_nodes):
                    if W[j ,k] == 0:
                        continue
                    noise_vector = np.random.normal(0, sigma_cor, size=num_dim)
                    cor_noise[:, j] += noise_vector
                    if not uncorrelate:
                        cor_noise[:, k] += -noise_vector
                    else:
                        noise_vector_2 = np.random.normal(0, sigma_cor, size=num_dim)
                        cor_noise[:, k] += noise_vector_2

        X_iter = X_iter - gamma * (grad.T + noise + cor_noise)
        X_iter = X_iter.dot(W)
        errors.append(consensus_distance(X_iter, A, B))
    return errors, X_iter


def optimize_GT(X, topology, A, B, gamma, sigma, num_iter=100, is_lyapunov=False):
    # X.shape = (num_dim, num_nodes)
    # A.shape = (num_nodes, num_dim, num_dim)
    # B.shape = (num_nodes, num_dim)
    def calculate_grad(A, B, X_iter, sigma):
        AXmB = (np.einsum("ijk,ik->ij", A, X_iter.T) - B) # shape (num_nodes, num_dim)
        grad = np.einsum("ijk,ij->ik", A, AXmB) # shape (num_nodes, num_dim)
        noise = np.random.normal(0, np.sqrt(sigma / num_dim), size=X.shape)
        return grad.T + noise

    
    num_dim, num_nodes = X.shape
    X_iter = np.copy(X)
    errors = [consensus_distance(X_iter, A, B)]
    Y = calculate_grad(A, B, X_iter, sigma)
    # Y = np.zeros_like(X_iter)
    grad = np.copy(Y)
    def lyapunov_f(X, Y):
        num_nodes = X.shape[1]
        x_mean = np.mean(X, axis=1)
        dist = [np.linalg.norm(X_iter[:,i] - x_mean) ** 2 for i in range(0, num_nodes)]
        xmxbar = np.mean(dist)
        y_mean = np.mean(Y, axis=1)
        dist = [np.linalg.norm(Y[:,i] - y_mean) ** 2 for i in range(0, num_nodes)]
        ymybar = np.mean(dist)
        return (xmxbar, ymybar)

    lyapunov = [lyapunov_f(X_iter, Y)]
    for i in range(0, num_iter):
        W = topology(i)
        prev_grad = np.copy(grad)
        grad = calculate_grad(A, B, X_iter, sigma)
        X_iter = (X_iter - gamma * Y).dot(W)
        Y = Y.dot(W) + grad - prev_grad
        # print(X_iter, np.mean(X_iter))
        errors += [consensus_distance(X_iter, A, B)]
        lyapunov += [lyapunov_f(X_iter, Y)]
    
    if is_lyapunov:
        return np.array(lyapunov).T, errors, X_iter
    return errors, X_iter


def optimize_D2(X, topology, A, B, gamma, sigma, num_iter=100):
    # X.shape = (num_dim, num_nodes)
    # A.shape = (num_nodes, num_dim, num_dim)
    # B.shape = (num_nodes, num_dim)
    
    num_dim, num_nodes = X.shape
    X_iter = np.copy(X)
    errors = [consensus_distance(X_iter, A, B)]
    prev_grad = np.zeros_like(X)
    prev_X = np.copy(X_iter)
    for i in range(0, num_iter):
        W = topology(i)
        AXmB = (np.einsum("ijk,ik->ij", A, X_iter.T) - B) # shape (num_nodes, num_dim)
        grad = np.einsum("ijk,ij->ik", A, AXmB) # shape (num_nodes, num_dim)
        noise = np.random.normal(0, np.sqrt(sigma / num_dim), size=X.shape)

        X_plus = 2 * X_iter - prev_X  - gamma * (grad.T + noise) + gamma * prev_grad
        # if i == 0, then this is equivalent to the standard SGD step X - gamma grad
        prev_grad = np.copy(grad.T + noise)
        prev_X = np.copy(X_iter)
        X_iter = X_plus.dot(W)
        errors.append(consensus_distance(X_iter, A, B))
    return errors, X_iter

class Optimizer:
    def __init__(self, X, topology, num_iter):
        self.X = X
        self.topology = topology
        self.num_iter = num_iter
    
    def optimize(self, method, params):
        pass

