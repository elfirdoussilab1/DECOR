from utils import *
import numpy as np
import pandas as pd
from csv import writer
from utils import plotting, dp_account, topology, optimizers
import matplotlib.pyplot as plt
import misc

if __name__ == "__main__":
    
    params = {
        "topology_names": ["centralized", "ring", "grid"],
        "gamma": 1.668e-3,
        "num_nodes": 16,
        "num_dim": 10,
        "c_clip":1.,
        "num_iter": 3500,
        "num_gossip": 1,
        "delta": 1e-5
    }

    
    A, B = generate_functions(params["num_nodes"], params["num_dim"], zeta = 0)
    epsilon_grid = np.array([1, 3, 5, 7, 10, 15, 20, 25, 30, 40]) # there is also 1 (but not intersting)
    # Storing sigmas and sigmas_cor for loss in function of epsilon
    sigmas = np.zeros((len(params['topology_names']), len(epsilon_grid))) 
    sigmas_cor = np.zeros((len(params['topology_names']), len(epsilon_grid))) 
    for j, target_eps in enumerate(epsilon_grid):
        for i, topology_name in enumerate(params['topology_names']):
            filename= f"result_gridsearch_{params['topology_names'][i]}_Corr_epsilon_{target_eps}.csv"
            df = pd.read_csv(filename)
            sigmas[i, j] = df.iloc[-1]["sigma"]
            sigmas_cor[i, j] = df.iloc[-1]["sigma_cor"]
            print("done")
        #plotting.plot_comparison_loss_CI(A = A, B = B, target_eps = target_eps, sigmas = sigmas[:,j], sigmas_cor = sigmas_cor[:,j],**params)
    plotting.loss_epsilon(epsilon_grid= epsilon_grid, A = A, B = B, sigmas = sigmas, sigmas_cor = sigmas_cor, **params)

    """
    seeds = np.arange(1, 6)
    # Plotting the results in a 3x3 plot (3, 5, 7 |10, 15, 20 | 25, 30, 40)
    epsilon_grid = epsilon_grid.reshape(3, 2)
    plt.style.use('ggplot')
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    W_centr = topology.FixedMixingMatrix("centralized", params["num_nodes"])
    
    for i in range(3):
        for j in range(2):
            X = np.ones(shape=(params["num_dim"], params["num_nodes"]))
            # sigma_cdp and sigma_ldp
            eps_iter = dp_account.reverse_eps(epsilon_grid[i, j], params["num_iter"], params["delta"])
            sigma_ldp = params["c_clip"] * np.sqrt(2 / eps_iter)
            sigma_cdp = sigma_ldp / np.sqrt(params["num_nodes"])

            # CDP
            errors_centr = []
            for seed in seeds:
                misc.fix_seed(seed)
                errors_centr.append(optimizers.optimize_decentralized_correlated(X, W_centr, A, B, params["gamma"], sigma_cdp, 0, params["c_clip"], num_gossip=params["num_gossip"], 
                num_iter=params["num_iter"])[0])

            t = np.arange(0, params["num_iter"] + 1)[::20]
            axes[i, j].semilogy(t, np.mean(errors_centr, axis = 0)[::20], color='tab:purple', linestyle = 'solid', alpha = 0.8)
            axes[i, j].fill_between(t, (np.mean(errors_centr, axis = 0) - np.std(errors_centr, axis = 0))[::20], (np.mean(errors_centr, axis = 0) + np.std(errors_centr, axis = 0))[::20], 
                    facecolor = 'tab:purple', alpha = 0.2)

            for k, topology_name in enumerate(params["topology_names"]):
                W = topology.FixedMixingMatrix(topology_name, params["num_nodes"])

                # Storing results
                errors_cor = []
                errors_ldp = []
                for seed in seeds:
                    misc.fix_seed(seed) 
                    filename= f"result_gridsearch_{topology_name}_Corr_epsilon_{epsilon_grid[i,j]}.csv"
                    df = pd.read_csv(filename)
                    sigma = df.iloc[-1]["sigma"]
                    sigma_cor = df.iloc[-1]["sigma_cor"]
                    errors_cor.append(optimizers.optimize_decentralized_correlated(X, W, A, B, params["gamma"], sigma, sigma_cor, params["c_clip"], num_gossip=params["num_gossip"], 
                    num_iter=params["num_iter"])[0])
                    errors_ldp.append(optimizers.optimize_decentralized_correlated(X, W, A, B, params["gamma"], sigma_ldp, 0, params["c_clip"], num_gossip=params["num_gossip"], 
                    num_iter=params["num_iter"])[0])

                axes[i, j].semilogy(t, np.mean(errors_cor, axis = 0)[::20], color = 'tab:green', 
                            linestyle = plotting.topo_to_style[topology_name], alpha = 0.8)
                axes[i, j].fill_between(t, (np.mean(errors_cor, axis = 0) - np.std(errors_cor, axis = 0))[::20], (np.mean(errors_cor, axis = 0) + np.std(errors_cor, axis = 0))[::20], 
                                facecolor = 'tab:green', alpha = 0.2)
                axes[i, j].semilogy(t, np.mean(errors_ldp, axis = 0)[::20], label=f"LDP with {topology_name}", color = 'tab:orange', 
                            linestyle = plotting.topo_to_style[topology_name], alpha = 0.8)
                axes[i, j].fill_between(t, (np.mean(errors_ldp, axis = 0) - np.std(errors_ldp, axis = 0))[::20], (np.mean(errors_ldp, axis = 0) + np.std(errors_ldp, axis = 0))[::20], 
                                facecolor = 'tab:orange', alpha = 0.2)
                axes[i, j].set_xlabel('iteration')
                #axes[i, j].set_ylabel('Loss')
                axes[i, j].set_title(f"User-privacy  {round(epsilon_grid[i, j])}")
                axes[i, j].grid(True)

    # Special Legend
    legend_hanles = []
    legend_hanles.append(plt.Line2D([], [], label='Algorithm', linestyle = 'None'))
    legend_hanles.append(plt.Line2D([], [], label='CDP', color = 'tab:purple'))
    legend_hanles.append(plt.Line2D([], [], label='Correlated-DSGD', color = 'tab:green'))
    legend_hanles.append(plt.Line2D([], [], label='LDP', color = 'tab:orange'))
    legend_hanles.append(plt.Line2D([], [], label='Topology', linestyle = 'None'))
    legend_hanles.append(plt.Line2D([], [], label='Centralized', linestyle = plotting.topo_to_style['centralized'], color = 'k'))
    legend_hanles.append(plt.Line2D([], [], label='Grid', linestyle = plotting.topo_to_style['grid'], color = 'k'))
    legend_hanles.append(plt.Line2D([], [], label='Ring', linestyle = plotting.topo_to_style['ring'], color = 'k'))

    # Make legend at the end of each row
    for i in range(3):
        axes[i, 0].set_ylabel('Loss')

    fig.legend(handles = legend_hanles, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=2, prop={'size': 15})
    folder_path = './losses'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    fig.savefig(folder_path + '/loss-n_{}-d_{}-lr_{}-clip_{}-delta_{}-T_{}.png'.format(params['num_nodes'], params['num_dim'], params['gamma'], 
        params['c_clip'], params['delta'], params['num_iter']), bbox_inches='tight')    
    """
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
#             errors_central = run_quadratics(sigma_cdp= , sigma_cor=0, lr=lr, clip=c_clip,
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
