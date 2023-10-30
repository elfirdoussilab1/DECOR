# Grid search for the best hyperparameters
import numpy as np
from utils import *
import pandas as pd
from utils import plotting

if __name__ == "__main__":
    params = {
            "gamma_grid": [4e-3],
            "num_nodes": 64,
            "num_dim": 10,
            "c_clip_grid":[1],
            "max_loss": 1e-2,
            "num_iter": 1500,
            "num_gossip": 1,
            "delta": 1e-5
        }
    
    A, B = generate_functions(params["num_nodes"], params["num_dim"], zeta = 0)
    eps_targets = [10, 15, 20, 25, 30, 40, 50]
    for eps_target in eps_targets:
        result = plotting.find_best_params(A, B, target_eps= eps_target, **params)
        filename= f"result_grid_epsilon_{eps_target}.csv"
        result.to_csv(filename)
        df = pd.read_csv(filename)
        # Plotting results
        for index, row in df.iterrows():
            if index == 0:
                continue
            gamma = row["gamma"]
            c_clip = row["c_clip"]
            sigma_cdp = row['sigma_cdp']
            sigma_cor = row['sigma_cor']
            plotting.plot_comparison_loss(A, B, gamma, params["num_nodes"], params["num_dim"], sigma_cdp, sigma_cor, c_clip, target_eps = eps_target)