# Grid search for the best hyperparameters
import numpy as np
from utils import *
import pandas as pd
from utils import plotting

if __name__ == "__main__":
    params = {
            "gamma_grid": np.logspace(-4, -2, 6),
            "num_nodes": 64,
            "num_dim": 10,
            "c_clip_grid":np.linspace(1, 10, 10),
            "target_eps": 10,
            "max_loss": 1e-2,
            "num_iter": 1000,
            "num_gossip": 1,
            "delta": 1e-4
        }
        
    A, B = generate_functions(params["num_nodes"], params["num_dim"], zeta = 0)
    result = plotting.find_best_params(A, B, **params)
    filename= "result_grid.csv"
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
        plotting.plot_comparison_loss(A, B, gamma, 64, 10, sigma_cdp, sigma_cor, c_clip)