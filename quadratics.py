from optimizers import *
from topology import *
from dp_account import *
from utils import *
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

if __name__ == "__main__":
    params = {
        "num_nodes": [4, 16, 256],
        "sigma_cdp": np.sqrt([1, 10, 100]),
        "sigma_cor": np.sqrt([10, 100, 1000]),
        "topology": ['ring', 'grid'],
        "num_dim": 25,
        "non_iid": 0
    }

    for n in params["num_nodes"]:
        A, B = generate_functions(n, params["num_dim"], params["non_iid"])
        for sigma_cdp in params["sigma_cdp"]:
            for topo_name in params["topology"]:
                plot_sigmacor_loss(A, B, sigma_cdp=sigma_cdp, num_nodes=n, topo_name=topo_name)

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
