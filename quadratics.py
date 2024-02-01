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

    folder = "./quadratics_for_n_16/numerical_results_for_n_16_and_clip_1/"
    A, B = generate_functions(params["num_nodes"], params["num_dim"], zeta = 1)
    epsilon_grid = np.array([1, 3, 5, 7, 10, 15, 20, 25, 30, 40]) # there is also 1 (but not intersting)
    # Storing sigmas and sigmas_cor for loss in function of epsilon
    sigmas = np.zeros((len(params['topology_names']), len(epsilon_grid))) 
    sigmas_cor = np.zeros((len(params['topology_names']), len(epsilon_grid))) 
    for j, target_eps in enumerate(epsilon_grid):
        for i, topology_name in enumerate(params['topology_names']):
            filename= folder + f"result_gridsearch_{params['topology_names'][i]}_Corr_epsilon_{target_eps}.csv"
            df = pd.read_csv(filename)
            sigmas[i, j] = df.iloc[-1]["sigma"]
            sigmas_cor[i, j] = df.iloc[-1]["sigma_cor"]
            print("done")
        #plotting.plot_comparison_loss_CI(A = A, B = B, target_eps = target_eps, sigmas = sigmas[:,j], sigmas_cor = sigmas_cor[:,j],**params)
    plotting.loss_epsilon(epsilon_grid= epsilon_grid, A = A, B = B, sigmas = sigmas, sigmas_cor = sigmas_cor, **params)