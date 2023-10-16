import numpy as np
import scipy.linalg as spl
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
from utils.topology import *
import time
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.cm as cm


def rdp_account(sigmacdp, sigmacor, clip, degree_matrix, adjacency_matrix, precision=0, p=0, sparse=True, cho=False):
    n = degree_matrix.shape[0]
    eps = 0

    if precision == 0:
        laplacian = degree_matrix - adjacency_matrix
        sigma_matrix = (sigmacdp ** 2) * np.eye(n) + (sigmacor ** 2) * laplacian
        if sparse:
            sigma_matrix = sps.csr_matrix(sigma_matrix)
        elif cho:
            L, low = spl.cho_factor(sigma_matrix)
        # print(L, low)

        max_entry = 0
        for i in range(n):
            # print(i)
            b = [int(j == i) for j in range(n)]
            if not sparse:
                x = spl.solve(sigma_matrix, b, assume_a='pos')
            elif cho:
                x = spl.cho_solve((L, low), b)
            else:
                x = spsl.spsolve(sigma_matrix, b)
            # print(b)
            # print(L @ x)
            # max_entry = max(max_entry, x @ x)
            max_entry = max(max_entry, x[i])

        eps = max_entry

    else:
        if sparse:
            degree_matrix = sps.csr_matrix(degree_matrix)
            dmax = max(np.diag(degree_matrix.todense()))
            # inv_sq_lambda = sps.csr_matrix(inv_sq_lambda)
        else:
            dmax = max(np.diag(degree_matrix))
        q = int(np.ceil(np.log(2 * np.sqrt(n) * clip ** 2 / (precision * sigmacdp ** 2)) / np.log(1 + sigmacdp ** 2 / (dmax * sigmacor ** 2))))
        if p == 0:
            p = int(np.ceil(np.sqrt(q)))
        print(q, q/p)
        lambda_matrix = (sigmacdp ** 2) * np.eye(n) + (sigmacor ** 2) * degree_matrix
        inv_sq_lambda = np.sqrt(np.diag(1 / np.diag(lambda_matrix)))
        M = (sigmacor ** 2) * inv_sq_lambda @ adjacency_matrix @ inv_sq_lambda
        M = sps.csr_matrix(M) if sparse else M
        if sparse:
            series_M, M_k = sps.csr_matrix(np.eye(n)), sps.csr_matrix(np.eye(n))
        else:
            series_M, M_k = np.eye(n), np.eye(n)
        if q > 1:
            for i in range(min(q-1, p-1)):
                # print(i)
                M_k = M_k @ M
                series_M += M

            if q > p:
                if sparse:
                    series_M_q, M_k_q = sps.csr_matrix(np.eye(n)), M_k
                else:
                    series_M_q, M_k_q = np.eye(n), M_k
                for i in range(int(np.ceil(q/p))):
                    # print(i)
                    M_k_q = M_k_q @ M_k
                    series_M_q += M_k_q
                series_M = series_M @ series_M_q

        inv_sigma_hat = inv_sq_lambda @ series_M @ inv_sq_lambda
        eps = max(np.diag(inv_sigma_hat))

    return 2 * (clip ** 2) * eps + precision


def rdp_compose_convert(num_iter, rdp_eps, delta):
    """The single-iteration RDP guaranteed is assumed to be of the form (alpha, alpha*rdp_eps)"""
    return num_iter * rdp_eps + 2 * np.sqrt(num_iter * rdp_eps * np.log(1 / delta))

def test_time():
    n, d = 10, 2
    sigmacdp, sigmacor = np.sqrt(1 / n), np.sqrt(100)
    clip = .1
    degree_matrix = d * np.eye(n)
    degree_matrix = sps.csr_matrix(degree_matrix)

    adjacency_matrix = np.zeros((n, n))
    np.fill_diagonal(adjacency_matrix[1:], 1, wrap=False)
    np.fill_diagonal(adjacency_matrix[:, 1:], 1, wrap=False)
    adjacency_matrix[0, n - 1] = 1
    adjacency_matrix[n - 1, 0] = 1
    adjacency_matrix = sps.csr_matrix(adjacency_matrix)

    precision = 1

    start_time = time.time()
    print(rdp_account(sigmacdp, sigmacor, clip, degree_matrix, adjacency_matrix, precision=0, sparse=False, cho=True))
    end_time_0 = time.time()
    print("time 0: {}".format(end_time_0 - start_time))
    print(rdp_account(sigmacdp, sigmacor, clip, degree_matrix, adjacency_matrix, precision=0, sparse=False))
    end_time_1 = time.time()
    print("time 1: {}".format(end_time_1 - end_time_0))
    print(rdp_account(sigmacdp, sigmacor, clip, degree_matrix, adjacency_matrix, precision=0, sparse=True))
    end_time_2 = time.time()
    print("time 2: {}".format(end_time_2 - end_time_1))
    print(
        rdp_account(sigmacdp, sigmacor, clip, degree_matrix, adjacency_matrix, precision=precision, p=0, sparse=False))
    end_time_3 = time.time()
    print("time 3: {}".format(end_time_3 - end_time_2))
    # print(rdp_account(sigmacdp, sigmacor, clip, degree_matrix, adjacency_matrix, precision=precision, p=0, sparse=True))
    # print("time 4: {}".format(time.time() - end_time_3))

def plot_pareto(c_clip=1, num_nodes=16, topo_name="ring"):
    topo = FixedMixingMatrix(topo_name, num_nodes)
    adjacency_matrix = np.array(topo(0) != 0, dtype=float)
    adjacency_matrix = adjacency_matrix - np.diag(np.diag(adjacency_matrix))
    degree_matrix = np.diag(adjacency_matrix @ np.ones_like(adjacency_matrix[0]))
    # print(adjacency_matrix)
    # print(degree_matrix)

    delta = 0.1
    # x = np.arange(0.01, 100, delta)
    # y = np.arange(0.01, 100, delta)
    # x = np.logspace(-2, 1, 100)
    # y = np.logspace(-2, 1, 100)
    # print(x)
    x = np.linspace(1e-2, 1e1, 100)
    y = np.linspace(1e-2, 1e1, 100)
    X, Y = np.meshgrid(x, y)
    # print(len(x))
    # print(np.exp(-X**2 - Y**2).shape)
    # couples = np.vstack([X.ravel(), Y.ravel()]).T
    Z = np.zeros_like(X)
    for i in range(len(x)):
        if i % 5 == 0:
            print("{}/{}".format(i, len(x)))
        for j in range(len(Y)):
            # Z[i, j] = rdp_account(1/np.sqrt(X[i, j]), 1/np.sqrt(Y[i, j]), c_clip, degree_matrix, adjacency_matrix)
            Z[i, j] = rdp_account(np.sqrt(X[i, j]), np.sqrt(Y[i, j]), c_clip, degree_matrix, adjacency_matrix)

    print(Z)
    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, Z, locator=ticker.LogLocator(subs='all'))
    # CS = ax.contour(X, Y, Z, levels=25)
    ax.clabel(CS, inline=True, fontsize=10)
    ax.set_title('Single-iteration RDP epsilon')
    ax.set_xlabel('$\sigma_{\mathrm{cdp}}^2$')
    ax.set_ylabel('$\sigma_{\mathrm{cor}}^2$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.savefig('plots/paretobis-n{}-{}.png'.format(num_nodes, topo_name))
    # plt.plot()
    # plt.show()

def plot_sigmacor(sigma_cdp= 0.1,c_clip=1, num_nodes=256, topo_name="ring"):
    topo = FixedMixingMatrix(topo_name, num_nodes)
    adjacency_matrix = np.array(topo(0) != 0, dtype=float)
    adjacency_matrix = adjacency_matrix - np.diag(np.diag(adjacency_matrix))
    degree_matrix = np.diag(adjacency_matrix @ np.ones_like(adjacency_matrix[0]))
    # print(adjacency_matrix)
    # print(degree_matrix)

    # delta = 0.01
    # x = np.arange(0.001, 10, delta)
    x = np.logspace(-3, 7, 50)
    print(len(x))
    # print(np.exp(-X**2 - Y**2).shape)
    # couples = np.vstack([X.ravel(), Y.ravel()]).T
    y = np.array([rdp_account(sigma_cdp, np.sqrt(xx), c_clip, degree_matrix, adjacency_matrix) for xx in x])

    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xscale('log')
    ax.set_yscale('log')
    # ax.set_ylim(0, 210)
    ax.set_title('Single-iteration RDP epsilon')
    ax.set_xlabel('$\sigma_{\mathrm{cor}}^2$')
    ax.set_ylabel('$\\varepsilon_{\mathrm{RDP}}$')
    plt.savefig('plots/sigmacor-n{}-sigmacdp{}-{}.png'.format(num_nodes,sigma_cdp,topo_name))
    # plt.plot()
    # plt.show()


if __name__ == "__main__":
    params = {
        "num_nodes": [4, 16, 256],
        "sigma_cdp": np.sqrt([1, 10, 100]),
        "sigma_cor": np.sqrt([10, 100, 1000]),
        "topology": ['ring', 'grid'],
    }
    # plot_pareto()
    # for n in params["num_nodes"]:
    #     for sigma_cdp in params["sigma_cdp"]:
    #         # for sigma_cor in params["sigma_cor"]:
    #             for topo_name in params["topology"]:
    #                 plot_sigmacor(sigma_cdp=sigma_cdp, num_nodes=n, topo_name=topo_name)

    # for n in params["num_nodes"]:
    #     for topo_name in params["topology"]:
    #         plot_pareto(num_nodes=n, topo_name=topo_name)
