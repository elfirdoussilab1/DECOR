# Grid search for the best hyperparameters
import numpy as np
from utils import *

if __name__ == "__main__":
    params = {
            "gamma_grid": np.logspace(np.log10(1e-6), np.log10(1e-2), 100),
            "num_nodes": 64,
            "num_dim": 10,
            "c_clip_grid":np.linspace(1, 100, 100),
            "sigma_cor_grid":np.linspace(1, 100, 100) ,
            "target_eps": 1e-2,
            "max_loss": 1e-2,
            "num_iter": 1000,
            "num_gossip": 1
        }
        
    A, B = generate_functions(64, 10, zeta = 0)
    filename= "result.csv"
    df = pd.read_csv(filename)
    # Plotting results
    for index, row in df.iterrows():
        if index == 0:
            continue
        gamma = row["gamma"]
        c_clip = row["c_clip"]
        sigma_cdp = row['sigma_cdp']
        sigma_cor = row['sigma_cor']
        plot_comparison_loss(A, B, gamma, 64, 10, sigma_cdp, sigma_cor, c_clip)