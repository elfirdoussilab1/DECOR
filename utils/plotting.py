import numpy as np
import misc
from utils import topology, dp_account, optimizers
import matplotlib.pyplot as plt
import os, math
import pandas as pd

# Plot single loss
def plot_loss(topology_name, method, A, B, gamma, num_nodes, num_dim, sigma, sigma_cor, c_clip, target_eps, num_gossip=1, num_iter = 1000, delta = 1e-4, seeds= [1, 2, 3, 4, 5]):
    X = np.ones(shape=(num_dim, num_nodes))
    W = topology.FixedMixingMatrix(topology_name, num_nodes)

    # Storing results
    errors = []
    for seed in seeds:
        misc.fix_seed(seed)
        errors.append(optimizers.optimize_decentralized_correlated(X, W, A, B, gamma, sigma, sigma_cor, c_clip, num_gossip=num_gossip, num_iter=num_iter)[0])
    eps = target_eps
    if not math.isclose(sigma_cor, 0): #Corr
        # Privacy 
        adjacency_matrix = np.array(W(0) != 0, dtype=float)
        adjacency_matrix = adjacency_matrix - np.diag(np.diag(adjacency_matrix))
        degree_matrix = np.diag(adjacency_matrix @ np.ones_like(adjacency_matrix[0]))
        eps_iter = dp_account.rdp_account(sigma, sigma_cor, c_clip, degree_matrix, adjacency_matrix)
        eps = dp_account.rdp_compose_convert(num_iter, eps_iter, delta)
    
    fig, ax = plt.subplots()
    t = np.arange(0, num_iter + 1)
    ax.semilogy(t, np.mean(errors, axis = 0), label=method)
    ax.fill_between(t, np.mean(errors, axis = 0) - np.std(errors, axis = 0), np.mean(errors, axis = 0) + np.std(errors, axis = 0), alpha = 0.3)
    ax.set_xlabel('iteration')
    ax.set_ylabel('loss')
    ax.set_title(f"loss with user-privacy  {round(eps)}")
    ax.legend() 
    
    folder_path = f"./plot-{topology_name}-{method}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plt.savefig(folder_path + '/loss-topo_{}-method_{}-n_{}-d_{}-lr_{}-clip_{}-sigma_{}-sigmacor_{}-delta_{}.png'.format(topology_name, method, num_nodes, num_dim, gamma, c_clip, round(sigma, 2) , round(sigma_cor, 2), delta))


# dictionary for plot colors ans style
topo_to_style = {"ring": (0, (1, 10)), "grid": (5, (10, 3)), "centralized": 'solid'}
method_to_marker = {"LDP": "^", "CDP": "s", "Corr": "o"}

# Function to plot threee losses: LDP, CD-SGD and CDP
def plot_comparison_loss(topology_name, A, B, num_nodes, num_dim, gamma, c_clip, sigma, sigma_cor, num_gossip=1, num_iter = 3000, delta = 1e-5, target_eps= None):
    """
    This function plots the comparison between CDP, CD-SGD and LDP
    Args:
        topology_name (str): topo name for LDP and Corr
        A (array): parameter A
        B (array): parameter B
        num_nodes (int): number of nodes
        num_dim (int): number of parameters for each worker
        gamma (float): learning rate
        c_clip (float): Gradient clip 
        sigma (float): standard deviations for Correlated noise
        sigma_cor (float): standard deviation of correlated noise
        num_gossip (int): gossip
        num_iter (int): total number of iterations

    """
    X = np.ones(shape=(num_dim, num_nodes))
    W = topology.FixedMixingMatrix(topology_name, num_nodes)
    W_centr = topology.FixedMixingMatrix("centralized", num_nodes)

    # fixing the seed
    misc.fix_seed(1)

    # sigma_cdp and sigma_ldp
    eps_iter = dp_account.reverse_eps(target_eps, num_iter, delta)
    sigma_ldp = c_clip * np.sqrt(2 / eps_iter)
    sigma_cdp = sigma_ldp / np.sqrt(num_nodes)

    # Learning
    errors_centr, _ = optimizers.optimize_decentralized_correlated(X, W_centr, A, B, gamma, sigma_cdp, 0, c_clip, num_gossip=num_gossip, num_iter=num_iter)
    errors_cor, _ = optimizers.optimize_decentralized_correlated(X, W, A, B, gamma, sigma, sigma_cor, c_clip, num_gossip=num_gossip, num_iter=num_iter)
    errors_ldp, _ = optimizers.optimize_decentralized_correlated(X, W, A, B, gamma, sigma_ldp, 0, c_clip, num_gossip=num_gossip, num_iter=num_iter)

    plt.style.use('ggplot')
    fig, ax = plt.subplots()
    ax.semilogy(errors_centr, label="CDP", color='tab:purple', linestyle = 'solid')
    ax.semilogy(errors_cor, label=f"CD-SGD with {topology_name}", color = 'tab:green', linestyle = topo_to_style[topology_name], marker = topo_to_marker[topology_name])
    ax.semilogy(errors_ldp, label="LDP", color = 'tab:orange', linestyle = topo_to_style[topology_name])
    ax.set_xlabel('iteration')
    ax.set_ylabel('loss')
    ax.set_title(f"loss with user-privacy  {target_eps}")
    ax.grid(True)
    ax.legend()
    
    folder_path = f'./comparison_losses'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plt.savefig(folder_path + '/loss-n_{}-d_{}-sigmacdp_{}-sigmaldp_{}-sigma_{}-sigma_cor_{}-delta_{}-T_{}.png'.format(num_nodes, num_dim, round(sigma_cdp, 2) , round(sigma_ldp, 2), round(sigma, 2), round(sigma_cor, 2), delta, num_iter))

# Plotting comparison of losses with confidence interval
def plot_comparison_loss_CI(topology_names, A, B, num_nodes, num_dim, gamma, c_clip, sigmas, sigmas_cor, target_eps, num_gossip=1, num_iter = 2500, delta = 1e-5, seeds= np.arange(1, 4)):

    X = np.ones(shape=(num_dim, num_nodes))
    W_centr = topology.FixedMixingMatrix("centralized", num_nodes)

    # sigma_cdp and sigma_ldp
    eps_iter = dp_account.reverse_eps(target_eps, num_iter, delta)
    sigma_ldp = c_clip * np.sqrt(2 / eps_iter)
    sigma_cdp = sigma_ldp / np.sqrt(num_nodes)

    # CDP
    errors_centr = []
    for seed in seeds:
        misc.fix_seed(seed)
        errors_centr.append(optimizers.optimize_decentralized_correlated(X, W_centr, A, B, gamma, sigma_cdp, 0, c_clip, num_gossip=num_gossip, num_iter=num_iter)[0])
    
    plt.style.use('ggplot')
    fig, ax = plt.subplots()
    fig.set_size_inches(3 * 2.54, 2 * 2.54)
    t = np.arange(0, num_iter + 1)
    ax.semilogy(t, np.mean(errors_centr, axis = 0), label="CDP", color='tab:purple', linestyle = 'solid')
    ax.fill_between(t, np.mean(errors_centr, axis = 0) - np.std(errors_centr, axis = 0), np.mean(errors_centr, axis = 0) + np.std(errors_centr, axis = 0), alpha = 0.3)

    for i, topology_name in enumerate(topology_names):
        W = topology.FixedMixingMatrix(topology_name, num_nodes)

        # Storing results
        errors_cor = []
        errors_ldp = []
        for seed in seeds: 
            errors_cor.append(optimizers.optimize_decentralized_correlated(X, W, A, B, gamma, sigmas[i], sigmas_cor[i], c_clip, num_gossip=num_gossip, num_iter=num_iter)[0])
            errors_ldp.append(optimizers.optimize_decentralized_correlated(X, W, A, B, gamma, sigma_ldp, 0, c_clip, num_gossip=num_gossip, num_iter=num_iter)[0])

        ax.semilogy(t, np.mean(errors_cor, axis = 0), label=f"CD-SGD with {topology_name}", color = 'tab:green', 
                    linestyle = topo_to_style[topology_name])
        ax.fill_between(t, np.mean(errors_cor, axis = 0) - np.std(errors_cor, axis = 0), np.mean(errors_cor, axis = 0) + np.std(errors_cor, axis = 0), alpha = 0.3)
        ax.semilogy(t, np.mean(errors_ldp, axis = 0), label=f"LDP with {topology_name}", color = 'tab:orange', linestyle = topo_to_style[topology_name])
        ax.fill_between(t, np.mean(errors_ldp, axis = 0) - np.std(errors_ldp, axis = 0), np.mean(errors_ldp, axis = 0) + np.std(errors_ldp, axis = 0), alpha = 0.3)
    
    
    ax.set_xlabel('iteration')
    ax.set_ylabel('loss')
    ax.set_title(f"loss with user-privacy  {round(target_eps)}")
    ax.grid(True)
    ax.legend(loc='upper left')
    
    folder_path = './comparison_losses_CI'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plt.savefig('comparison_losses_CI/loss-n_{}-d_{}-lr_{}-clip_{}-sigma-cdp_{}-sigmas_ldp{}-delta_{}.png'.format(num_nodes, num_dim, gamma, c_clip, round(sigma_cdp, 2), round(sigma_ldp, 2), delta))

# Plotting loss in function of epsilon
def loss_epsilon(topology_names, epsilon_grid, A, B, num_nodes, num_dim, gamma, c_clip, sigmas, sigmas_cor, num_gossip=1, num_iter = 2500, delta = 1e-5, seeds= np.arange(1, 4)):
    """
    Plotting Losses in function of epsilons
    args:  
        epsilon_grid (list): list of epsilons that we consider
        A (array): parameter A
        B (array): parameter B
        num_nodes (int): number of nodes
        num_dim (int): number of parameters for each worker
        gammas (list of float): learning rates : at index 0 for CDP, 1 for Corr and 2 for LDP
        c_clips (list float): Gradient clip (same as gammas)
        sigmas (2D list of floats): shape (len(topology_names), len(epsilon_grod))
        sigma_cor (2D list of floats): shape (len(topology_names), len(epsilon_grod))
        num_gossip (int): gossip
        num_iter (int): total number of iterations
    """
    X = np.ones(shape=(num_dim, num_nodes))
    W_centr = topology.FixedMixingMatrix("centralized", num_nodes)

    # Init figure
    plt.style.use('ggplot')
    fig, ax = plt.subplots()
    fig.set_size_inches(3 * 2.54, 2 * 2.54)

    # List that will contain loss for each epsilon
    errors_centr = [] # shape (len(seeds), len(epsilon_grid))

    # CDP
    for seed in seeds:
        misc.fix_seed(seed)
        losses_centr = []
        for target_eps in epsilon_grid:
            eps_iter = dp_account.reverse_eps(target_eps, num_iter, delta)
            sigma_cdp = c_clip * np.sqrt(2 / eps_iter) / np.sqrt(num_nodes)
            losses_centr.append(np.mean(optimizers.optimize_decentralized_correlated(X, W_centr, A, B, gamma, sigma_cdp, 0, c_clip, num_gossip=num_gossip, num_iter=num_iter)[0][-200:-1]))
        # Adding result
        errors_centr.append(losses_centr)

    # Plotting CDP
    ax.semilogy(epsilon_grid, np.mean(errors_centr, axis = 0), label="CDP", color='tab:purple', 
                linestyle = 'solid', marker = "D")
    ax.fill_between(epsilon_grid, np.mean(errors_centr, axis = 0) - np.std(errors_centr, axis = 0), np.mean(errors_centr, axis = 0) + np.std(errors_centr, axis = 0), alpha = 0.3)

    # Corr and LDP
    for i, topology_name in enumerate(topology_names):
        errors_cor = []
        errors_ldp = []
        W = topology.FixedMixingMatrix(topology_name, num_nodes)

        for seed in seeds:
            misc.fix_seed(seed)
            losses_cor = []
            losses_ldp = []
            for j, target_eps in enumerate(epsilon_grid):
                # sigma_cdp and sigma_ldp
                eps_iter = dp_account.reverse_eps(target_eps, num_iter, delta)
                sigma_ldp = c_clip * np.sqrt(2 / eps_iter)
                losses_cor.append(np.mean(optimizers.optimize_decentralized_correlated(X, W, A, B, gamma, sigmas[i][j], sigmas_cor[i][j], c_clip, num_gossip=num_gossip, num_iter=num_iter)[0][-200:-1]))
                losses_ldp.append(np.mean(optimizers.optimize_decentralized_correlated(X, W, A, B, gamma, sigma_ldp, 0, c_clip, num_gossip=num_gossip, num_iter=num_iter)[0][-200:-1]))

            # Adding result
            errors_cor.append(losses_cor)
            errors_ldp.append(losses_ldp)
        # Plotting the corresponding result
        ax.semilogy(epsilon_grid, np.mean(errors_cor, axis = 0), label=f"CD-SGD with {topology_name}", color = 'tab:green', 
                    linestyle = topo_to_style[topology_name], marker = "o")
        ax.fill_between(epsilon_grid, np.mean(errors_cor, axis = 0) - np.std(errors_cor, axis = 0), np.mean(errors_cor, axis = 0) + np.std(errors_cor, axis = 0), alpha = 0.3)

        ax.semilogy(epsilon_grid, np.mean(errors_ldp, axis = 0), label=f"LDP with {topology_name}", color = 'tab:orange', 
                    linestyle = topo_to_style[topology_name], marker = "^")
        ax.fill_between(epsilon_grid, np.mean(errors_ldp, axis = 0) - np.std(errors_ldp, axis = 0), np.mean(errors_ldp, axis = 0) + np.std(errors_ldp, axis = 0), alpha = 0.3)
    

    ax.set_xlabel('User-Privacy $\epsilon$')
    ax.set_ylabel('error')
    ax.set_title(f"Evolution of L2 Loss with User-privacy ")
    ax.grid(True)
    ax.legend(loc='upper left')
    
    folder_path = './loss_epsilon'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plt.savefig(folder_path + '/loss-n_{}-d_{}-topology_{}-lr_{}-clip_{}-delta_{}-T_{}.png'.format(num_nodes, num_dim, topology_name, gamma, c_clip, delta, num_iter))

# Plotting loss in function of Number of Nodes
def loss_num_nodes(topology_name, A, B, num_nodes_grid, num_dim, gamma, c_clip, sigma, sigma_cor, target_eps, num_gossip=1, num_iter = 300, delta = 1e-5, seeds = np.arange(1, 6)):
    pass

#------------------------------------------------------------------------------------------------------------------------------#
# Parameter tuning
# Binary search for 'sigma_cor' in a grid 'sigma_cor_grid' such that the privacy is close close to the target 'eps_taget'
def find_sigma_cor(sigma_cdp, sigma_cor_grid, c_clip, degree_matrix, adjacency_matrix, eps_target):
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
    eps_end = dp_account.rdp_account(sigma_cdp, sigma_cor_grid[-1], c_clip, degree_matrix, adjacency_matrix)
    eps = dp_account.rdp_account(sigma_cdp, sigma_cor, c_clip, degree_matrix, adjacency_matrix)
    if eps_end > eps_target: # No hope, since the function is monotonous (epsilon non-increasing with sigma_cor)
        return []
    
    if eps - eps_target < 1e-4 and eps > eps_target: # found
        return [sigma_cor]
    elif eps > eps_target: # increase sigma
        return find_sigma_cor(sigma_cdp, sigma_cor_grid[n // 2 :], c_clip, degree_matrix, adjacency_matrix, eps_target)
    else: #eps < eps_target
        return find_sigma_cor(sigma_cdp, sigma_cor_grid[:n // 2], c_clip, degree_matrix, adjacency_matrix, eps_target)

        
# Function to find the best hyperparameters (gamma, clip, sigma_cdp, sigma_cor) in terms of loss for a fixed privacy 'target_eps'
def find_best_params(topology_name, method, A, B, num_nodes, num_dim, gamma_grid, c_clip_grid, max_loss, target_eps, delta, num_gossip=1, num_iter= 1500):
    """
    This function searches for the best parameters sigma_cdp and sigma_cor 
    such that they give a good performance (loss_cdp < min_loss) and a privacy under target_eps

    Args:
        method (str): either CDP, LDP or CD-SGD 
        topology_name (str): topology's name
        A (array): parameter A
        B (array): parameter B
        num_nodes (int): number of nodes
        num_dim (int): number of parameters for each worker
        gamma_grid (list): grid of learning rates
        c_clip_grid (list): grid of clipping thresholds
        max_loss (float): maximum loss method
        target_eps (float): maximum user-privacy epsilon
        delta (float): privacy parameter
        num_gossip (int): gossip (default 1),
        num_iter (int): total number of iterations (default 1000)
    
        Return:
            result (Pandas.DataFrame)

    """
    misc.fix_seed(1) # for reproducibility
    X = np.ones(shape=(num_dim, num_nodes))
    W = topology.FixedMixingMatrix(topology_name, num_nodes)

    # Privacy 
    adjacency_matrix = np.array(W(0) != 0, dtype=float)
    adjacency_matrix = adjacency_matrix - np.diag(np.diag(adjacency_matrix))
    degree_matrix = np.diag(adjacency_matrix @ np.ones_like(adjacency_matrix[0]))

    # Epsilon_iter 
    target_eps_iter = dp_account.reverse_eps(target_eps, num_iter, delta)

    # Initialization
    data = [{"topology": topology_name, "method": method, "gamma": gamma_grid[0], "c_clip": c_clip_grid[0], "eps_iter": target_eps_iter, "eps": target_eps, "sigma": -1, "sigma_cor": -1, "loss":np.inf}]
    result = pd.DataFrame(data)
    
    # Searching
    for gamma in gamma_grid:
        print(f"looping for gamma {gamma}")

        for c_clip in c_clip_grid:
            print(f"looping for clip {c_clip}")

            # Determining sigma_cdp and sigma_ldp
            sigma_ldp = c_clip * np.sqrt(2/target_eps_iter)
            sigma_cdp = sigma_ldp/np.sqrt(num_nodes)
            
            if "Corr" in method:
                # Looking for sigma_cor for which dp_account = target_eps
                sigma_grid = np.linspace(sigma_cdp, sigma_ldp, 50)
                sigma_cor_grid = np.linspace(1, 1000, 1000)
                for sigma in sigma_grid:
                    all_sigma_cor = find_sigma_cor(sigma, sigma_cor_grid, c_clip, degree_matrix, adjacency_matrix, target_eps_iter)
                    print(f"sigma_cor {all_sigma_cor}")
                    # Now test on the loss condition
                    if len(all_sigma_cor) != 0: # Not empty
                        for sigma_cor in all_sigma_cor:
                            errors, _ = optimizers.optimize_decentralized_correlated(X, W, A, B, gamma, sigma, sigma_cor, c_clip, num_gossip=num_gossip, num_iter=num_iter)

                            if np.mean(errors[-200:-1])  <= max_loss and result["loss"].iloc[-1] - np.mean(errors[-200:-1]) >= 0:
                                eps_iter = dp_account.rdp_account(sigma, sigma_cor, c_clip, degree_matrix, adjacency_matrix)
                                new_row = {"topology": topology_name,
                                            "method": method,
                                            "gamma": gamma, 
                                            "c_clip": c_clip,
                                            "eps_iter": eps_iter ,
                                            "eps": dp_account.rdp_compose_convert(num_iter, eps_iter, delta),
                                            "sigma": sigma,
                                            "sigma_cor": sigma_cor,
                                            "loss": np.mean(errors[-200:-1]),
                                            
                                            }
                                result = pd.concat([result, pd.DataFrame([new_row])], ignore_index=True)
                                #result = result.append(new_row, ignore_index = True)
                                print(f"added with privacy {new_row['eps']}")

                
            else: # LDP or CDP
                if "CDP" in method:
                    sigma = sigma_cdp
                    
                else:
                    sigma = sigma_ldp
                    
                errors, _ = optimizers.optimize_decentralized_correlated(X, W, A, B, gamma, sigma, 0, c_clip, num_gossip=num_gossip, num_iter=num_iter)
                if np.mean(errors[-200:-1])  <= max_loss and result["loss"].iloc[-1] - np.mean(errors[-200:-1]) > 0:
                    new_row = {"topology": topology_name,
                                "method": method,
                                "gamma": gamma, 
                                "c_clip": c_clip,
                                "eps_iter": target_eps_iter ,
                                "eps": target_eps,
                                "sigma": sigma,
                                "sigma_cor": 0,
                                "loss": np.mean(errors[-200:-1]),
                                
                                }
                    result = pd.concat([result, pd.DataFrame([new_row])], ignore_index=True)
                    #result = result.append(new_row, ignore_index = True)
                    print(f"added with privacy {new_row['eps']}")

    return result

#------------------------------------------------------------------------------------------------------------------------------#
# Old but Gold
# Function to plot loss in function of sigma_cor
def plot_sigmacor_loss(A, B, sigma_cdp= 0.1, c_clip=1, lr=0.1, num_gossip=1, num_nodes=256, topo_name="ring"):
    topo = topology.FixedMixingMatrix(topo_name, num_nodes)
    adjacency_matrix = np.array(topo(0) != 0, dtype=float)
    adjacency_matrix = adjacency_matrix - np.diag(np.diag(adjacency_matrix))
    degree_matrix = np.diag(adjacency_matrix @ np.ones_like(adjacency_matrix[0]))
    # print(adjacency_matrix)
    # print(degree_matrix)

    # delta = 0.01
    # x = np.arange(0.001, 10, delta)
    x = np.logspace(-3, 7, 50)
    print(len(x))
    # print(np.exp(-X**2 - Y**2).shape)
    # couples = np.vstack([X.ravel(), Y.ravel()]).T
    y = np.array([min(run_optimize(sigma_cdp=sigma_cdp, sigma_cor=np.sqrt(xx), lr=lr, clip=c_clip, num_nodes=num_nodes,
                                         num_gossip=num_gossip, A=A, B=B, topology=topo_name, num_iter=1000, seed=123, non_iid=0, num_dim=25)) for xx in x])

    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xscale('log')
    ax.set_yscale('log')
    # ax.set_ylim(0, 210)
    # ax.set_title('Best loss')
    ax.set_xlabel('$\sigma_{\mathrm{cor}}^2$')
    ax.set_ylabel('Best loss')
    plt.savefig('plots/loss-sigmacor-n{}-sigmacdp{}-{}.png'.format(num_nodes,sigma_cdp,topo_name))
    # plt.plot()
    # plt.show()
    