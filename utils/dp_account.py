import numpy as np
import scipy.linalg as spl
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
from utils.topology import *
import time, math
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.cm as cm
from utils import param_search
import warnings
import pandas as pd

def rdp_account(sigmacdp, sigmacor, clip, degree_matrix, adjacency_matrix, precision=0, p=0,
                sparse=True, cho=False):
    """Returns the output of Algorithm 2 in the paper (single-iteration RDP without amplification by subsampling)
    TODO: Redirect all calls of rdp_account in the project to rdp_compose_convert
    """
    n = degree_matrix.shape[0]
    eps = 0

    if precision == 0:
        laplacian = degree_matrix - adjacency_matrix
        sigma_matrix = (sigmacdp ** 2) * np.eye(n) + (sigmacor ** 2) * laplacian
        if sparse:
            sigma_matrix = sps.csr_matrix(sigma_matrix)
        elif cho:
            L, low = spl.cho_factor(sigma_matrix)

        max_entry = 0
        for i in range(n):
            b = [int(j == i) for j in range(n)]
            if not sparse:
                x = spl.solve(sigma_matrix, b, assume_a='pos')
            elif cho:
                x = spl.cho_solve((L, low), b)
            else:
                x = spsl.spsolve(sigma_matrix, b)
            max_entry = max(max_entry, x[i])

        eps = max_entry

    else:
        if sparse:
            degree_matrix = sps.csr_matrix(degree_matrix)
            dmax = max(np.diag(degree_matrix.todense()))
        else:
            print(degree_matrix)
            print(np.diag(degree_matrix))
            dmax = max(np.diag(degree_matrix))
        q = int(np.ceil(np.log(2 * np.sqrt(n) * clip ** 2 / (precision * sigmacdp ** 2)) / np.log(
            1 + sigmacdp ** 2 / (dmax * sigmacor ** 2))))
        if p == 0:
            p = int(np.ceil(np.sqrt(q)))
        print(q, q / p)
        lambda_matrix = (sigmacdp ** 2) * np.eye(n) + (sigmacor ** 2) * degree_matrix
        inv_sq_lambda = np.sqrt(np.diag(1 / np.diag(lambda_matrix)))
        M = (sigmacor ** 2) * inv_sq_lambda @ adjacency_matrix @ inv_sq_lambda
        M = sps.csr_matrix(M) if sparse else M
        if sparse:
            series_M, M_k = sps.csr_matrix(np.eye(n)), sps.csr_matrix(np.eye(n))
        else:
            series_M, M_k = np.eye(n), np.eye(n)
        if q > 1:
            for i in range(min(q - 1, p - 1)):
                M_k = M_k @ M
                series_M += M

            if q > p:
                if sparse:
                    series_M_q, M_k_q = sps.csr_matrix(np.eye(n)), M_k
                else:
                    series_M_q, M_k_q = np.eye(n), M_k
                for i in range(int(np.ceil(q / p))):
                    M_k_q = M_k_q @ M_k
                    series_M_q += M_k_q
                series_M = series_M @ series_M_q

        inv_sigma_hat = inv_sq_lambda @ series_M @ inv_sq_lambda
        eps = max(np.diag(inv_sigma_hat))

    return 2 * (clip ** 2) * eps + precision

def user_level_rdp(num_iter, eps_iter, delta):
    return num_iter * eps_iter + 2 * np.sqrt(num_iter * eps_iter * np.log(1 / delta))

def minimize_alpha(rdp_eps, alpha_int_max=100, n_points=1000):
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    alpha_int_space = np.arange(2, alpha_int_max + 1, 1)
    argmin_int = np.nanargmin([rdp_eps(alpha_int) for alpha_int in alpha_int_space])
    alpha_int_min = alpha_int_space[argmin_int]

    alpha_lower = alpha_int_min - 1. + 1e-4
    alpha_upper = alpha_int_min + 1.
    alpha_float_space = np.linspace(alpha_lower, alpha_upper, n_points)

    return min([rdp_eps(alpha_float) for alpha_float in alpha_float_space])


def rdp_compose_convert(num_iter, delta, sigmacdp, sigmacor, clip, degree_matrix, adjacency_matrix, subsample=1., batch_size=1.):
    """

    Main Args:
        subsample: subsampling ratio, set to 1 for user-level DP
        batch_size: mini-batch size, ignored if user-level DP

    Returns: DP epsilon after T iterations of Correlated DSGD; user-level if subsample == 1. and example-level otherwise.

    """
    if math.isclose(subsample, 1.):
        rdp_eps_no_sub = rdp_account(sigmacdp, sigmacor, clip, degree_matrix, adjacency_matrix)
        return user_level_rdp(num_iter, rdp_eps_no_sub, delta)

    rdp_eps_no_sub = rdp_account(sigmacdp, sigmacor, clip/batch_size, degree_matrix, adjacency_matrix)
    rdp_eps_func = lambda alpha: alpha * rdp_eps_no_sub
    rdp_eps_func_sub = rdp_subsample(rdp_eps_func, subsample)
    out = minimize_alpha(rdp_eps_func_sub)
    return out


def reverse_eps(eps, num_iter, delta, clip, degree_matrix, adjacency_matrix, subsample=1., batch_size=1.):
    """ Find single-iteration RDP epsilon (eps_iter) from total DP epsilon (eps)
    IMPORTANT: only works assuming user-level DP, or no data subsampling
    TODO: implement case subsample < 1
    """
    if math.isclose(subsample, 1.):
        return (np.sqrt(np.log(1 / delta) + eps) - np.sqrt(np.log(1 / delta))) ** 2 / num_iter

    else: # Binary search
        # Find couples (sigma, sigma_corr) that give eps
        sigma_grid = np.linspace(1e-3, 1e-1, 100)
        sigma_cor_grid = np.linspace(1e-3, 1e-1, 1000)
        
        # Initialize dataframe
        data = [{"clip": clip, "sigma": -1, "sigma-cor": -1, "eps_iter": -1, "eps": -1}]
        result = pd.DataFrame(data)
        for sigma in sigma_grid:
            print(f"Looping for sigma {sigma}")
            all_sigma_cor = param_search.find_sigma_cor(eps, sigma, sigma_cor_grid, clip, degree_matrix, adjacency_matrix, num_iter, delta, subsample, batch_size)
            print(all_sigma_cor)
            if len(all_sigma_cor) != 0: # Not empty
                for sigma_cor in all_sigma_cor:
                        
                    eps_iter = rdp_account(sigma, sigma_cor, clip, degree_matrix, adjacency_matrix)
                    new_row = {"c_clip": clip,
                                "sigma": sigma, 
                                "sigma-cor": sigma_cor,
                                "eps_iter": eps_iter ,
                                "eps": rdp_compose_convert(num_iter, delta, sigma, sigma_cor, clip, degree_matrix, adjacency_matrix, subsample, batch_size)
                                }
                    result = pd.concat([result, pd.DataFrame([new_row])], ignore_index=True)
                    #result = result.append(new_row, ignore_index = True)
                    print(f"added with privacy {new_row['eps']}")

        return result


def rdp_subsample(eps, subsample):
    """
    Args:
        eps: function of alpha, returns RDP epsilon (with alpha)
        subsample: subsampling ratio

    Returns: rdp subsampled epsilon as a function of alpha
    """

    def int_rdp_subsample(alpha):
        return (1. / (alpha - 1.)) * np.log(1 + (subsample ** 2) * math.comb(alpha, 2) * min(4 * (np.exp(eps(2)) - 1), 2 * np.exp(eps(2))) + 2 * sum([(subsample ** j) * math.comb(alpha, j) * np.exp((j - 1) * eps(j)) for j in range(3, alpha + 1)]))

    def out(alpha):
        tmp = 0. if math.floor(alpha) == 1 else (1-alpha+math.floor(alpha)) * (math.floor(alpha)-1) / (alpha-1) * int_rdp_subsample(math.floor(alpha))
        return tmp + (alpha-math.floor(alpha)) * (math.ceil(alpha)-1) / (alpha - 1) * int_rdp_subsample(math.ceil(alpha))

    return out


def test_time():
    n, d = 10, 2
    sigmacdp, sigmacor = np.sqrt(1 / n), np.sqrt(100)
    clip = .1
    degree_matrix = d * np.eye(n)
    # degree_matrix = sps.csr_matrix(degree_matrix)

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
    print(rdp_compose_convert(10, 1e-5, sigmacdp, sigmacor, clip, degree_matrix, adjacency_matrix, subsample=.5, batch_size=1))
    end_time_3 = time.time()
    print("time 3: {}".format(end_time_3 - end_time_2))
    # print(
    #     rdp_account(sigmacdp, sigmacor, clip, degree_matrix, adjacency_matrix, precision=precision, p=0, sparse=False))
    # end_time_3 = time.time()
    # print("time 3: {}".format(end_time_3 - end_time_2))
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


def plot_sigmacor(sigma_cdp=0.1, c_clip=1, num_nodes=256, topo_name="ring"):
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
    plt.savefig('plots/sigmacor-n{}-sigmacdp{}-{}.png'.format(num_nodes, sigma_cdp, topo_name))
    # plt.plot()
    # plt.show()


if __name__ == "__main__":
    test_time()
    # params = {
    #     "num_nodes": [4, 16, 256],
    #     "sigma_cdp": np.sqrt([1, 10, 100]),
    #     "sigma_cor": np.sqrt([10, 100, 1000]),
    #     "topology": ['ring', 'grid'],
    # }
    # plot_pareto()
    # for n in params["num_nodes"]:
    #     for sigma_cdp in params["sigma_cdp"]:
    #         # for sigma_cor in params["sigma_cor"]:
    #             for topo_name in params["topology"]:
    #                 plot_sigmacor(sigma_cdp=sigma_cdp, num_nodes=n, topo_name=topo_name)

    # for n in params["num_nodes"]:
    #     for topo_name in params["topology"]:
    #         plot_pareto(num_nodes=n, topo_name=topo_name)
