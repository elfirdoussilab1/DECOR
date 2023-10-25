from utils import *
import numpy as np
import pandas as pd
import json, os, itertools
from csv import writer
from utils import plotting, dp_account



if __name__ == "__main__":
    
    params = {
        "gamma": 3.981e-3,
        "num_nodes": 64,
        "num_dim": 10,
        "sigma_cdp":12.44,
        "sigma_cor": 44.31,
        "c_clip":1,
        "num_iter": 1000,
        "num_gossip": 1,
        "delta": 1e-4
    }

    
    A, B = generate_functions(64, 10, zeta = 0)
    plotting.plot_comparison_loss_CI(A, B, **params)

    
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
