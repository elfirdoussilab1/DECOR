import numpy as np


# The functions below generate two arrays:
#   A of shape (num_nodes, num_dim, num_dim)
#   B of shape (num_nodes, num_dim) (like every node vector)

def generate_functions(num_nodes, num_dim, zeta):
    A = [1 / np.sqrt(num_nodes) * np.eye(num_dim) * (i + 1) for i in range(num_nodes)]
    B = [np.random.normal(0, np.sqrt(zeta) / (i + 1), size=num_dim) for i in range(num_nodes)]
    # B = [np.ones(num_dim) * np.sqrt(zeta) / (i + 1) for i in range(0, num_nodes)]
    return np.array(A), np.array(B)

def generate_functions_for_DGD(num_nodes, num_dim):

    A = [np.random.normal(size=(num_dim, num_dim)) for i in range(num_nodes)]
    B = [np.zeros(num_dim) for i in range(num_nodes)]
    # B = [np.ones(num_dim) * np.sqrt(zeta) / (i + 1) for i in range(0, num_nodes)]
    return np.array(A), np.array(B)

def generate_functions_for_GT(num_nodes, num_dim):
    # f_i = || A_ix - b_i ||_2^2
    A = [np.random.normal(size=(num_dim, num_dim)) for i in range(0, num_nodes)]
    B = [np.random.normal(size=num_dim) for i in range(0, num_nodes)]
    return np.array(A), np.array(B)

def generate_consensus_functions(num_nodes, num_dim):
    # f_i = || A_ix - b_i ||_2^2
    A = [np.identity(num_dim) for i in range(0, num_nodes)]
    B = [1 + i + np.random.normal(size=num_dim) for i in range(0, num_nodes)]
    return np.array(A), np.array(B)

def consensus_distance(X, A, B): # ||x-x*||^2
    # X.shape = (num_dim, num_nodes)

    x_star = np.linalg.inv(np.einsum("ikj,ikl->jl", A, A)).dot(np.einsum("ijk,ij->k", A, B))
    num_nodes = X.shape[1]
    dist = [np.linalg.norm(X[:,i] - x_star) ** 2 for i in range(num_nodes)]
    return np.mean(dist)

def grid_search(gamma_grid, optimize, target_accuracy):
    positions = []
    all_errors = []
    best_error = np.inf
    backup_index = 0 # if not reached target accuracy - choose the best one reached
    for i, gamma in enumerate(gamma_grid):
        errors, _ = optimize(gamma)
        all_errors.append(errors)
        errors = np.array(errors)
        errors[np.isnan(errors)] = np.inf
        if np.min(errors) < best_error: # for the backup
            best_error = np.min(errors)
            backup_index = i
        try:
            first_pos = np.nonzero(errors < target_accuracy)[0][0]
            positions.append(first_pos)
        except:
            positions.append(np.inf)

    if np.isinf(np.min(positions)): # target accuracy never reached
        print('\033[93m' + "target accuracy not reached" + '\033[0m')
        print("overall best error achieved:", best_error, "for gamma:", gamma_grid[backup_index])
        return all_errors[backup_index], gamma_grid[backup_index], np.inf
    best_index = np.argmin(positions)
    print("best num of iterations:", np.min(positions), "for gamma:", gamma_grid[best_index])
    print("overall best error achieved:", best_error)
    return all_errors[best_index], gamma_grid[best_index], np.min(positions)


def grid_search_two_params(gamma_grid, alpha_grid, optimize, target_accuracy):
    positions = []
    all_errors = {}
    best_error = np.inf
    backup_index = 0 # if not reached target accuracy - choose the best one reached
    indexes = []
    for i, gamma in enumerate(gamma_grid):
        for j, alpha in enumerate(alpha_grid):
            errors, _ = optimize(gamma, alpha)
            all_errors[(i, j)] = errors
            errors = np.array(errors)
            errors[np.isnan(errors)] = np.inf
            indexes += [(i, j)]
            if np.min(errors) < best_error: # for the backup
                best_error = np.min(errors)
                backup_index = (i, j)
            try:
                first_pos = np.nonzero(np.array(errors) < target_accuracy)[0][0]
                positions.append(first_pos)
            except:
                positions.append(np.inf)

    if np.isinf(np.min(positions)): # target accuracy never reached
        print('\033[93m' + "target accuracy not reached" + '\033[0m')
        print("overall best error achieved:", best_error, "for gamma:", gamma_grid[backup_index[0]],
                                                        ", alpha:", alpha_grid[backup_index[1]])
        return all_errors[backup_index], (gamma_grid[backup_index[0]], alpha_grid[backup_index[1]])
    best_index = indexes[np.argmin(positions)]
    print("best num of iterations:", np.min(positions), "for gamma:", gamma_grid[best_index[0]],
                                                        ", alpha:", alpha_grid[best_index[1]])
    print("overall best error achieved:", best_error)
    return all_errors[best_index], (gamma_grid[best_index[0]], alpha_grid[best_index[1]])


def experiment_DGD(W_new, gamma_grid):
    optimize = lambda gamma: optimize_decentralized(X, W_new, A, B, gamma, sigma, num_iter)
    errors, _, best_pos = grid_search(gamma_grid, optimize, target_acc)
    return best_pos

