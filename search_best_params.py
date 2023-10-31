# Grid search for the best hyperparameters
import numpy as np
from utils import *
import pandas as pd
from utils import plotting

if __name__ == "__main__":
    params_cdp = {
            "topology_name": "centralized",
            "method": "CDP",
            "gamma_grid": [4.6e-3],
            "num_nodes": 64,
            "num_dim": 10,
            "c_clip_grid":[1.],
            "max_loss": 1e-2,
            "num_iter": 3500,
            "num_gossip": 1,
            "delta": 1e-5
        }
    params_ldp = {
            "topology_name": "ring",
            "method": "LDP",
            "gamma_grid": [1.7e-3],
            "num_nodes": 64,
            "num_dim": 10,
            "c_clip_grid":[1.5],
            "max_loss": 10.,
            "num_iter": 3500,
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
            "num_iter": 3500,
            "num_gossip": 1,
            "delta": 1e-5
        }
    
    A, B = generate_functions(params_cdp["num_nodes"], params_cdp["num_dim"], zeta = 0)
    eps_targets = [10, 15, 20, 25, 30, 40, 50]
    # LDP
    for eps_target in eps_targets:
        result = plotting.find_best_params(A = A, B = B, target_eps= eps_target, **params_ldp)
        filename= f"result_grid_ldp_epsilon_{eps_target}.csv"
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
        plotting.plot_loss(row["topology"], row["method"], A, B, row["gamma"], params_ldp["num_nodes"], params_ldp["num_dim"], row['sigma'], row['sigma_cor'], row["c_clip"], 
                           target_eps = eps_target, num_iter = params_ldp["num_iter"], delta = params_ldp["delta"])
    # CDP
    
    for eps_target in eps_targets:
        result = plotting.find_best_params(A = A, B = B, target_eps= eps_target, **params_cdp)
        filename= f"result_grid_cdp_epsilon_{eps_target}.csv"
        result.to_csv(filename)
        df = pd.read_csv(filename)
        # Plotting best results
        row = df.iloc[-1] 
        plotting.plot_loss(row["topology"], row["method"], A, B, row["gamma"], params_cdp["num_nodes"], params_cdp["num_dim"], row['sigma'], row['sigma_cor'], row["c_clip"], 
                           target_eps = eps_target, num_iter = params_cdp["num_iter"], delta = params_cdp["delta"])
        
    # Corr
    for eps_target in eps_targets:
        result = plotting.find_best_params(A = A, B = B, target_eps= eps_target, **params_corr)
        filename= f"result_grid_corr_epsilon_{eps_target}.csv"
        result.to_csv(filename)
        df = pd.read_csv(filename)
        # Plotting best result
        row = df.iloc[-1] 
        plotting.plot_loss(row["topology"], row["method"], A, B, row["gamma"], params_corr["num_nodes"], params_corr["num_dim"], row['sigma'], row['sigma_cor'], row["c_clip"], 
                           target_eps = eps_target, num_iter = params_corr["num_iter"], delta = params_corr["delta"])
        