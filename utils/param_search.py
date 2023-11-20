# In this file, we implement useful functions to search for privacy parameters
import numpy as np
from . import dp_account

# Parameter searching
# Binary search for 'sigma_cor' in a grid 'sigma_cor_grid' such that the privacy is close close to the target 'eps_taget'
def find_sigma_cor(eps_target, sigma, sigma_cor_grid, clip, degree_matrix, adjacency_matrix, num_iter, delta, subsample = 1., batch_size = 1.):
    """
    This function aims to find values of sigma_cor for which we have a privacy less than eps_target
    """
    # median
    n = len(sigma_cor_grid)
    if n == 0:
        return []
    if n == 1: # single element
        return list(sigma_cor_grid)
    
    sigma_cor = sigma_cor_grid[ n // 2]
    eps_end = dp_account.rdp_compose_convert(num_iter, delta, sigma, sigma_cor_grid[-1], clip, degree_matrix, adjacency_matrix, subsample, batch_size)
    eps = dp_account.rdp_compose_convert(num_iter, delta, sigma, sigma_cor, clip, degree_matrix, adjacency_matrix, subsample, batch_size)
    if eps_end > eps_target: # No hope, since the function is monotonous (epsilon non-increasing with sigma_cor)
        print(f"eps_end {eps_end} is greater than target {eps_target}")
        return []
    
    if eps - eps_target < 1 and eps > eps_target: # found
        print(f"found {eps}")
        return [sigma_cor]
    elif eps > eps_target: # increase sigma. 
        return find_sigma_cor(eps_target, sigma, sigma_cor_grid[n // 2 :], clip, degree_matrix, adjacency_matrix, num_iter, delta, subsample, batch_size)
    else: #eps < eps_target  
        return find_sigma_cor(eps_target, sigma,  sigma_cor_grid[:n // 2], clip, degree_matrix, adjacency_matrix, num_iter, delta, subsample, batch_size)
