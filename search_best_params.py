# Grid search for the best hyperparameters
import numpy as np
from utils import *
import pandas as pd
from utils import plotting

if __name__ == "__main__":
    params = {
            "topology_name": "centralized",
            "method": "Corr",
            "gamma_grid": [1.7e-3],
            "num_nodes": 64,
            "num_dim": 10,
            "c_clip_grid":[1.5],
            "max_loss": 100,
            "num_iter": 2500,
            "num_gossip": 1,
            "delta": 1e-5
        }
    """
    params_ldp = {
            "topology_name": "ring",
            "method": "LDP",
            "gamma_grid": [1.7e-3],
            "num_nodes": 64,
            "num_dim": 10,
            "c_clip_grid":[1.5],
            "max_loss": 10.,
            "num_iter": 2500,
            "num_gossip": 1,
            "delta": 1e-5
        }
    params_corr = {
            "topology_name": "ring",
            "method": "Corr",
            "gamma_grid": [1.7e-3],
            "num_nodes": 64,
            "num_dim": 10,
            "c_clip_grid":[1.5],
            "max_loss": 1.,
            "num_iter": 2500,
            "num_gossip": 1,
            "delta": 1e-5
        }
    """  
    A, B = generate_functions(params["num_nodes"], params["num_dim"], zeta = 0)
    eps_targets = [1, 3, 5, 7, 10, 15, 20, 25, 30, 40]

    for eps_target in eps_targets:
        result = plotting.find_best_params(A = A, B = B, target_eps= eps_target, **params)
        filename= f"result_gridsearch_{params['topology_name']}_{params['method']}_epsilon_{eps_target}.csv"
        result.to_csv(filename)
        df = pd.read_csv(filename)
        # Plotting results
        """
        for index, row in df.iterrows():
            if index == 0:
                continue
            gamma = row["gamma"]
            c_clip = row["c_clip"]
            sigma = row['sigma']
            sigma_cor = row['sigma_cor']
            plotting.plot_loss(params_ldp["topology_name"], params_ldp["method"], A, B, gamma, params_ldp["num_nodes"], params_ldp["num_dim"], sigma, sigma_cor, c_clip, target_eps = eps_target,
                                num_iter = params_ldp["num_iter"], delta = params_ldp["delta"])
        """
        # Plotting best result
        row = df.iloc[-1] 
        plotting.plot_loss(row["topology"], row["method"], A, B, row["gamma"], params["num_nodes"], params["num_dim"], row['sigma'], row['sigma_cor'], row["c_clip"], 
                           target_eps = eps_target, num_iter = params["num_iter"], delta = params["delta"])
        