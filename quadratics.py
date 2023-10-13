from optimizers import *
from topology import *
from dp_account import *
from utils import *
import math
import pandas as pd
import json, os, itertools
from csv import writer


def run_quadratics(**inparams):
    np.random.seed(inparams["seed"])
    n, d = inparams["num_nodes"], inparams["num_dim"]
    A, B = inparams["A"], inparams["B"]
    X = np.ones(shape=(d, n))
    topo = FixedMixingMatrix(inparams["topology"], n)
    errors, _ = optimize_decentralized_correlated(X, topo, A, B, inparams["lr"], inparams["sigma_cdp"],
                                                  inparams["sigma_cor"], inparams["clip"],
                                                  num_gossip=inparams["num_gossip"], num_iter=inparams["num_iter"])

    return errors

def plot_sigmacor_loss(A, B, sigma_cdp= 0.1, c_clip=1, lr=0.1, num_gossip=1, num_nodes=256, topo_name="ring"):
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
    y = np.array([min(run_quadratics(sigma_cdp=sigma_cdp, sigma_cor=np.sqrt(xx), lr=lr, clip=c_clip, num_nodes=num_nodes,
                                         num_gossip=num_gossip, A=A, B=B, topology=topo_name, num_iter=1000, seed=123, non_iid=0, num_dim=25)) for xx in x])

    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xscale('log')
    ax.set_yscale('log')
    # ax.set_ylim(0, 210)
    # ax.set_title('Best loss')
    ax.set_xlabel('$\sigma_{\mathrm{cor}}^2$')
    ax.set_ylabel('Best loss')
    plt.savefig('plots/loss-sigmacor-n{}-sigmacdp{}-{}.png'.format(num_nodes,sigma_cdp,topo_name))
    # plt.plot()
    # plt.show()


def plot_comparison_loss(A, B, gamma, num_nodes, num_dim, sigma_cdp, sigma_cor, c_clip, num_gossip=1, num_iter = 1000):
    """
    This function plots the comparison between CDP, CD-SGD and LDP
    Args:
        A (array): parameter A
        B (array): parameter B
        gamma (float): learning rate
        num_nodes (int): number of nodes
        num_dim (int): number of parameters for each worker
        sigma_cdp (float): standard deviation of CDP noise
        sigma_cor (float): standard deviation of correlated noise
        c_clip (float): Gradient clip
        num_gossip (int): gossip
        num_iter (int): total number of iterations

    """
    X = np.ones(shape=(num_dim, num_nodes))
    W_ring = FixedMixingMatrix("ring", num_nodes)
    W_centr = FixedMixingMatrix("centralized", num_nodes)

    # Privacy 
    adjacency_matrix = np.array(W_ring(0) != 0, dtype=float)
    adjacency_matrix = adjacency_matrix - np.diag(np.diag(adjacency_matrix))
    degree_matrix = np.diag(adjacency_matrix @ np.ones_like(adjacency_matrix[0]))

    # eps_rdp_iteration = rdp_account(sigma, sigma_cor, c_clip, degree_matrix, adjacency_matrix, sparse=False, precision=0.1)
    eps_rdp_iteration = rdp_account(sigma_cdp, sigma_cor, c_clip, degree_matrix, adjacency_matrix)

    # Learning
    sigma_ldp = c_clip * np.sqrt(2/eps_rdp_iteration)
    errors_centr, _ = optimize_decentralized_correlated(X, W_centr, A, B, gamma, sigma_cdp, sigma_cor, c_clip, num_gossip=num_gossip, num_iter=num_iter)
    errors_cor, _ = optimize_decentralized_correlated(X, W_ring, A, B, gamma, sigma_cdp, sigma_cor, c_clip, num_gossip=num_gossip, num_iter=num_iter)
    errors_ldp, _ = optimize_decentralized_correlated(X, W_ring, A, B, gamma, sigma_ldp, 0, c_clip, num_gossip=num_gossip, num_iter=num_iter)

    fig, ax = plt.subplots()
    ax.semilogy(errors_centr, label="CDP")
    ax.semilogy(errors_cor, label="correlated DSGD")
    ax.semilogy(errors_ldp, label="LDP")
    ax.set_xlabel('iteration')
    ax.set_ylabel('loss')
    ax.set_title(f"loss with privacy eps per iteration {eps_rdp_iteration}")
    ax.legend()
    folder_path = './comparison_losses'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plt.savefig('comparison_losses/loss-n_{}-d_{}-lr_{}-clip_{}-sigmacdp_{}-sigmacor_{}-sigmaldp_{}.png'.format(num_nodes, num_dim, gamma, c_clip, round(sigma_cdp, 2) , round(sigma_cor, 2), round(sigma_ldp, 2)))


def find_sigma_cor(sigma_cdp, sigma_cor_grid, c_clip, degree_matrix, adjacency_matrix, eps_target):
    """
    This function aims to find values of sigma_cor for whch we have a privacy less than eps_target
    """
    # median
    n = len(sigma_cor_grid)
    if n == 0:
        return []
    if n == 1: # single element
        return list(sigma_cor_grid)
    sigma_cor = sigma_cor_grid[ n // 2]
    eps_end = rdp_account(sigma_cdp, sigma_cor_grid[-1], c_clip, degree_matrix, adjacency_matrix)
    eps = rdp_account(sigma_cdp, sigma_cor, c_clip, degree_matrix, adjacency_matrix)
    if eps_end > eps_target: # No hope, since the function is monotonous (epsilon non-increasing with sigma_cor)
        return []
    elif abs(eps - eps_target) < 1e-4: # found
        return [sigma_cor]
    elif eps > eps_target: # increase sigma
        return find_sigma_cor(sigma_cdp, sigma_cor_grid[n // 2 :], c_clip, degree_matrix, adjacency_matrix, eps_target)
    else: #eps < eps_target
        return find_sigma_cor(sigma_cdp, sigma_cor_grid[:n // 2], c_clip, degree_matrix, adjacency_matrix, eps_target)


        


def find_best_params(A, B, gamma, num_nodes, num_dim, max_loss, target_eps, c_clip, num_gossip=1, num_iter= 1000):
    """
    This function searches for the best parameters sigma_cdp and sigma_cor 
    such that they give a good performance (loss_cdp < min_loss) and a privacy under target_eps

    Args:
        A (array): parameter A
        B (array): parameter B
        gamma (float): learning rate
        num_nodes (int): number of nodes
        num_dim (int): number of parameters for each worker
        max_loss (float): maximum loss for CDp
        target_eps (float): maximum user-privacy
    
        Return:
            sigma_cdp, sigma_cor, eps, loss_cdp (dict)

    """
    X = np.ones(shape=(num_dim, num_nodes))
    W_ring = FixedMixingMatrix("ring", num_nodes)
    W_centr = FixedMixingMatrix("centralized", num_nodes)

    # Privacy 
    adjacency_matrix = np.array(W_ring(0) != 0, dtype=float)
    adjacency_matrix = adjacency_matrix - np.diag(np.diag(adjacency_matrix))
    degree_matrix = np.diag(adjacency_matrix @ np.ones_like(adjacency_matrix[0]))

    # sigma_ldp
    sigma_ldp = c_clip * np.sqrt(2/target_eps)
    sigma_cdp_grid = np.linspace(sigma_ldp/np.sqrt(num_nodes), sigma_ldp, 100)
    print(f"lower bound sigma cdp {sigma_ldp/np.sqrt(num_nodes)}")
    print(f"upper bound sigma cdp {sigma_ldp}")

    # Initialization
    eps_0 = rdp_account(sigma_cdp_grid[0], 0, c_clip, degree_matrix, adjacency_matrix)
    errors_centr, _ = optimize_decentralized_correlated(X, W_centr, A, B, gamma, sigma_cdp_grid[0], 0, c_clip, num_gossip=num_gossip, num_iter=num_iter)

    data = [{"sigma_cdp": sigma_ldp/np.sqrt(num_nodes), "sigma_cor": 0, "eps": eps_0, "loss_cdp":errors_centr[-1], "loss_cor": 10}]
    result = pd.DataFrame(data)

    # Searching
    for sigma_cdp in sigma_cdp_grid:
        print(f"loop for sigma_cdp {sigma_cdp}")
        # Looking for sigma_cor for which dp_account < target_eps
        sigma_cor_grid = np.linspace(1, 100, 1000)
        all_sigma_cor = find_sigma_cor(sigma_cdp, sigma_cor_grid, c_clip, degree_matrix, adjacency_matrix, target_eps)
        print(f"sigma_cor {all_sigma_cor}")
        # Now test on the loss condition
        if len(all_sigma_cor) != 0: # Not empty
            for sigma_cor in all_sigma_cor:
                errors_centr, _ = optimize_decentralized_correlated(X, W_centr, A, B, gamma, sigma_cdp, sigma_cor, c_clip, num_gossip=num_gossip, num_iter=num_iter)
                errors_cor, _ = optimize_decentralized_correlated(X, W_ring, A, B, gamma, sigma_cdp, sigma_cor, c_clip, num_gossip=num_gossip, num_iter=num_iter)

                if errors_centr[-1] <= max_loss and result["loss_cor"].iloc[-1] - np.mean(errors_cor[800:]) > 0:
                    new_row = {"sigma_cdp":sigma_cdp, 
                               "sigma_cor": sigma_cor, "eps": rdp_account(sigma_cdp, sigma_cor, c_clip, degree_matrix, adjacency_matrix),
                                "loss_cdp": errors_centr[-1],
                                 "loss_cor": errors_cor[-1] }
                    result = pd.concat([result, pd.DataFrame([new_row])], ignore_index=True)
                    #result = result.append(new_row, ignore_index = True)
                    print(f"added with privacy {rdp_account(sigma_cdp, sigma_cor, c_clip, degree_matrix, adjacency_matrix)}")

        else:
            continue
    return result


if __name__ == "__main__":
    params = {
        "num_nodes": 64,
        #"sigma_cdp": [10],
        #"sigma_cor": [12],
        "num_dim": 10,
        "gamma": 0.01,
        "c_clip":1,
        "non_iid": 0
    }
    
    A, B = generate_functions(params["num_nodes"], params["num_dim"], params["non_iid"])
    result = find_best_params(A, B, params["gamma"], params["num_nodes"], params["num_dim"], max_loss= 1e-2, target_eps=1e-2, c_clip=params["c_clip"],num_gossip=1, num_iter= 1000)
    filename= "result.csv"
    result.to_csv(filename, index=False)
    # Plotting results
    for index, row in result.iterrows():
        sigma_cdp = row['sigma_cdp']
        sigma_cor = row['sigma_cor']
        plot_comparison_loss(A, B, params["gamma"], params["num_nodes"], params["num_dim"], sigma_cdp, sigma_cor, params["c_clip"])


    #for n in params["num_nodes"]:
    #    A, B = generate_functions(n, params["num_dim"], params["non_iid"])
    #    for sigma_cdp in params["sigma_cdp"]:
    #        for sigma_cor in params["sigma_cor"]:
    #            plot_comparison_loss(A, B, params["gamma"], n, params["num_dim"], sigma_cdp, sigma_cor, params["c_clip"], num_iter=1000)

# if __name__ == "__main__":
#     base_params = {
#         "num_nodes": 100,
#         "num_dim": 100,
#         "non_iid": 0,
#         "num_iter": 1000,
#         "seed": 123,
#     }
#     # base_params = {
#     #     "num_nodes": 100,
#     #     "num_dim": 10,
#     #     "non_iid": 0,
#     #     "num_iter": 100,
#     #     "seed": 123,
#     # }
#
#     path = 'results-n{}-d{}-zeta{}-T{}'.format(*base_params.values())
#     while os.path.isdir(path):
#         path = '_' + path
#     os.mkdir(path)
#     os.chdir(path)
#
#     with open('base_params.json', mode='w') as f:
#         json.dump(base_params, f)
#
#     params = {
#         "sigma": np.sqrt([1, 10, 100]),
#         "sigma_cor": np.sqrt([10, 100, 1000]),
#         "topology": ['ring', 'grid'],
#         "c_clip": [0.5, 1, 2.5, 5],
#         "lr": [0.01, 0.1, 0.5],
#         "num_gossip": [1, 4]
#     }
#
#     # params = {
#     #     "sigma": np.sqrt([0.5]),
#     #     "sigma_cor": np.sqrt([10]),
#     #     "topology": ['ring'],
#     #     "c_clip": [1],
#     #     "lr": [0.1],
#     #     "num_gossip": [1]
#     # }
#
#     colnames = ['method'] + list(params.keys()) + ['step ' + str(i + 1) for i in range(base_params["num_iter"])]
#     for i in range(len(colnames)):
#         if colnames[i] == "sigma":
#             colnames[i] = "sigma^2"
#         if colnames[i] == "sigma_cor":
#             colnames[i] = "sigma_cor^2"
#     # df = pd.DataFrame(
#     #     columns=['method'] + list(params.keys()) + ['step ' + str(i + 1) for i in range(base_params["num_iter"])])
#     # df = df.rename(columns={"sigma": "sigma^2", "sigma_cor": "sigma_cor^2"})
#
#     A, B = generate_functions(base_params["num_nodes"], base_params["num_dim"], base_params["non_iid"])
#     total_exp = list(itertools.product(*params.values()))
#     num_exp = len(total_exp)
#
#     with open('numerical_results.csv', 'a') as f:
#         writer_object = writer(f)
#         writer_object.writerow(colnames)
#
#         for i, x in enumerate(total_exp):
#             if i % 5 == 0:
#                 print('Completed experiments: {}/{}'.format(i, num_exp))
#             sigma, sigma_cor, topo_name, c_clip, lr, num_gossip = x
#
#             topo = FixedMixingMatrix(topo_name, base_params["num_nodes"])
#             adjacency_matrix = np.array(topo(0) != 0, dtype=float)
#             adjacency_matrix = adjacency_matrix - np.diag(np.diag(adjacency_matrix))
#             degree_matrix = np.diag(adjacency_matrix @ np.ones_like(adjacency_matrix[0]))
#
#             # eps_rdp_iteration = rdp_account(sigma, sigma_cor, c_clip, degree_matrix, adjacency_matrix, sparse=False, precision=0.1)
#             eps_rdp_iteration = rdp_account(sigma, sigma_cor, c_clip, degree_matrix, adjacency_matrix)
#
#             sigma_ldp = c_clip * np.sqrt(2/eps_rdp_iteration)
#             # print("eps_rdp_iteration: {}".format(eps_rdp_iteration))
#             # print("sigmacdp: {}, sigmaldp: {}, sigmacor: {}".format(sigma, sigma_ldp, sigma_cor))
#
#             errors_corr = run_quadratics(sigma_cdp=sigma, sigma_cor=sigma_cor, lr=lr, clip=c_clip,
#                                          num_gossip=num_gossip, A=A, B=B, topology=topo_name, **base_params)
#             errors_ldp = run_quadratics(sigma_cdp=sigma_ldp, sigma_cor=0, lr=lr, clip=c_clip,
#                                         num_gossip=num_gossip, A=A, B=B, topology=topo_name, **base_params)
#             errors_central = run_quadratics(sigma_cdp=sigma, sigma_cor=0, lr=lr, clip=c_clip,
#                                             num_gossip=num_gossip, A=A, B=B, topology='centralized', **base_params)
#
#             delta = 1 / (base_params["num_nodes"] ** 1.1)
#             eps = [rdp_compose_convert(i + 1, eps_rdp_iteration, delta) for i in range(base_params["num_iter"])]
#
#             x = list(x)
#             x[0], x[1] = x[0]**2, x[1]**2
#             y, z = x.copy(), x.copy()
#             y[0], y[1], z[1] = sigma_ldp**2, 0, 0
#
#             writer_object.writerow(['Correlated'] + x + list(zip(errors_corr, eps)))
#             writer_object.writerow(['LDP'] + y + list(zip(errors_ldp, eps)))
#             writer_object.writerow(['Central'] + z + list(zip(errors_central, eps)))
#
#             # df.loc[len(df)] = ['Correlated'] + x + list(zip(errors_corr, eps))
#             # df.loc[len(df)] = ['LDP'] + y + list(zip(errors_ldp, eps))
#             # df.loc[len(df)] = ['Central'] + z + list(zip(errors_central, eps))
#
#         # df.to_csv('numerical_results.csv')
