# In this file, we implement useful functions to search for privacy parameters
import numpy as np
from . import dp_account
import pandas as pd

# Parameter searching
# Binary search for 'sigma_cor' in a grid 'sigma_cor_grid' such that the privacy is close close to the target 'eps_taget'
def find_sigma_cor_eps(eps_target, sigma, sigma_cor_grid, clip, degree_matrix, adjacency_matrix, num_iter, delta, subsample = 1., batch_size = 1.):
    """
    This function aims to find values of sigma_cor for which we have a privacy less than eps_target # CHECKED
    """
    # median
    n = len(sigma_cor_grid)
    if n == 0:
        return []
    if n == 1: # single element
        return list(sigma_cor_grid)
    
    sigma_cor = sigma_cor_grid[ n // 2]
    eps_end = dp_account.rdp_compose_convert(num_iter, delta, sigma, sigma_cor_grid[-1], clip, degree_matrix, adjacency_matrix, subsample, batch_size)
    eps_start = dp_account.rdp_compose_convert(num_iter, delta, sigma, sigma_cor_grid[0], clip, degree_matrix, adjacency_matrix, subsample, batch_size)
    eps = dp_account.rdp_compose_convert(num_iter, delta, sigma, sigma_cor, clip, degree_matrix, adjacency_matrix, subsample, batch_size)
    if eps_end > eps_target or eps_start < eps_target: # No hope, since the function is monotonous (epsilon non-increasing with sigma_cor)
        print(f"eps_target {eps_target} is not between end {eps_end} and start {eps_start}")
        return []
    
    if eps - eps_target < 0.05 and eps > eps_target: # found
        print(f"found {eps}")
        return [sigma_cor]
    elif eps > eps_target: # increase sigma. 
        return find_sigma_cor_eps(eps_target, sigma, sigma_cor_grid[n // 2 :], clip, degree_matrix, adjacency_matrix, num_iter, delta, subsample, batch_size)
    else: #eps < eps_target  
        return find_sigma_cor_eps(eps_target, sigma,  sigma_cor_grid[:n // 2], clip, degree_matrix, adjacency_matrix, num_iter, delta, subsample, batch_size)

def binary_search_eps(eps, num_iter, delta, num_nodes, clip, topology_name, degree_matrix, adjacency_matrix, subsample, batch_size, multiple = True):
    sigma_grid = np.linspace(clip /1000, clip/100 , 50)
    sigma_cor_grid = np.linspace(5*clip /10000, clip/10, 500)
    
    # Initialize dataframe
    result = pd.DataFrame(columns = ["clip", "sigma", "sigma-cor", "eps-iter", "eps", "sigma-cdp", "sigma-ldp"])
    for sigma in sigma_grid:
        print(f"Looping for sigma {sigma}")
        all_sigma_cor = find_sigma_cor_eps(eps, sigma, sigma_cor_grid, clip, degree_matrix, adjacency_matrix, num_iter, delta, subsample, batch_size)
        print(all_sigma_cor)
        if len(all_sigma_cor) != 0: # Not empty
            
            for sigma_cor in all_sigma_cor:
                    
                eps_iter = dp_account.rdp_account(sigma, sigma_cor, clip, degree_matrix, adjacency_matrix)
                sigma_ldp = clip * np.sqrt(2 / eps_iter)
                new_row = {"clip": clip,
                            "sigma": sigma, 
                            "sigma-cor": sigma_cor,
                            "eps-iter": eps_iter ,
                            "eps": dp_account.rdp_compose_convert(num_iter, delta, sigma, sigma_cor, clip, degree_matrix, adjacency_matrix, subsample, batch_size),
                            "sigma-cdp": sigma_ldp / num_nodes,
                            "sigma-ldp": sigma_ldp
                            }
                result = pd.concat([result, pd.DataFrame([new_row])], ignore_index=True)
                #result = result.append(new_row, ignore_index = True)
                print(f"added with privacy {new_row['eps']}")
                if not multiple:
                    if int(new_row['eps']) == eps: # if we found result
                        return result

    return result
