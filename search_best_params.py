# Grid search for the best hyperparameters
import numpy as np
from utils import *
import pandas as pd
from utils import plotting, dp_account, topology

if __name__ == "__main__":
    params = {
            "num_nodes": 16,
            "c_clip_grid":[1.],
            "max_loss": np.inf,
            "num_iter": 3500,
            "num_gossip": 1,
            "delta": 1e-5,
            "subsample": 64/3750,
            "batch-size": 64
        }
    """
    params_ldp = {
            "topology_name": "ring",
            "method": "LDP",
            "gamma_grid": [1.7e-3],
            "num_nodes": 16,
            "num_dim": 10,
            "c_clip_grid":[1.],
            "max_loss": 10.,
            "num_iter": 3500,
            "num_gossip": 1,
            "delta": 1e-5
        }
    params_corr = {
            "topology_name": "ring",
            "method": "Corr",
            "gamma_grid": [1.7e-3],
            "num_nodes": 16,
            "num_dim": 10,
            "c_clip_grid":[1.],
            "max_loss": 1.,
            "num_iter": 3500,
            "num_gossip": 1,
            "delta": 1e-5
        }
    """  

    # Parameters
    num_nodes = 16
    topology_name = "centralized"
    W = topology.FixedMixingMatrix("centralized", num_nodes)
    adjacency_matrix = np.array(W(0) != 0, dtype=float)
    adjacency_matrix = adjacency_matrix - np.diag(np.diag(adjacency_matrix))
    degree_matrix = np.diag(adjacency_matrix @ np.ones_like(adjacency_matrix[0]))

    clip = 2.
    delta = 1e-5
    num_iter = 1000
    subsample = 64/3750
    batch_size = 64

    eps = 5
    eps_iter = dp_account.reverse_eps(eps, num_iter, delta, num_nodes, clip, topology_name, degree_matrix, adjacency_matrix, subsample, batch_size, multiple = False)
    print(eps_iter)

    """
    # Plotting rdp_compose with (sigma, sigma_cor)
    sigma = 5e-3
    sigma_cor_grid = np.linspace(1e-2, 1, 100)

    eps = np.zeros_like(sigma_cor_grid)
    for i in range(len(eps)):
        eps[i] = dp_account.rdp_compose_convert(num_iter, delta, sigma, sigma_cor_grid[i],
                                                clip, degree_matrix, adjacency_matrix, subsample, batch_size)
    
    plt.semilogy(sigma_cor_grid, eps)
    plt.xlabel('$\sigma_{\mathrm{cor}}$')
    plt.ylabel('$\epsilon$')
    folder_path = './results-search-sigma'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plt.savefig(folder_path + f'/epsilon_vs_sigma-cor_sigma={sigma}.png', bbox_inches='tight')

   
    # Create a 3D plot
    sigma_mesh, sigma_cor_mesh = np.meshgrid(sigma_grid, sigma_cor_grid)
    eps = np.zeros_like(sigma_mesh)
    for i in range(len(sigma_grid)):
        for j in range(len(sigma_cor_grid)):
            eps[i, j] = dp_account.rdp_compose_convert(num_iter, delta, sigma_grid[i], sigma_cor_grid[j],
                                                clip, degree_matrix, adjacency_matrix)
            
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surf = ax.plot_surface(sigma_grid, sigma_cor_grid, eps, cmap='viridis')

    # Add color bar which maps values to colors
    fig.colorbar(surf)

    # Set labels and title
    ax.set_xlabel('sigma')
    ax.set_ylabel('sigma-cor')
    ax.set_zlabel('epsilon')

    folder_path = './results-tuning-mnist'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    fig.savefig(folder_path + '/loss_3D.png', bbox_inches='tight')

    #result = dp_account.reverse_eps(eps, num_iter, delta, clip, degree_matrix, adjacency_matrix, subsample, batch_size)
    #filename= f"result_gridsearch_example-level_{'centralized'}_epsilon_{eps}.csv"
    #result.to_csv(filename)

    A, B = generate_functions(params["num_nodes"], params["num_dim"], zeta = 0)
    eps_targets = [1, 3, 5, 7, 10, 15, 20, 25, 30, 40]
    
    topologies = ["centralized", "grid", "ring"]
    for topology_name in topologies:
        for eps_target in eps_targets:
            result = plotting.find_best_params(A = A, B = B, target_eps= eps_target, topology_name= topology_name, **params)
            filename= f"result_gridsearch_{topology_name}_{params['method']}_epsilon_{eps_target}.csv"
            result.to_csv(filename)
            df = pd.read_csv(filename)
            # Plotting best result
            row = df.iloc[-1] 
            plotting.plot_loss(row["topology"], row["method"], A, B, row["gamma"], params["num_nodes"], params["num_dim"], row['sigma'], row['sigma_cor'], row["c_clip"], 
                            target_eps = eps_target, num_iter = params["num_iter"], delta = params["delta"])
    """