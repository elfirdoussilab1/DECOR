import numpy as np
import topology, dp_account, optimizers
import matplotlib.pyplot as plt
import os

# Function to plot threee losses: LDP, CD-SGD and CDP
def plot_comparison_loss(A, B, gamma, num_nodes, num_dim, sigma_cdp, sigma_cor, c_clip, num_gossip=1, num_iter = 1000, target_eps = None):
    """
    This function plots the comparison between CDP, CD-SGD and LDP
    Args:
        A (array): parameter A
        B (array): parameter B
        gamma (float): learning rate
        num_nodes (int): number of nodes
        num_dim (int): number of parameters for each worker
        sigma_cdp (float): standard deviation of CDP noise
        sigma_cor (float): standard deviation of correlated noise
        c_clip (float): Gradient clip
        num_gossip (int): gossip
        num_iter (int): total number of iterations

    """
    X = np.ones(shape=(num_dim, num_nodes))
    W_ring = topology.FixedMixingMatrix("ring", num_nodes)
    W_centr = topology.FixedMixingMatrix("centralized", num_nodes)

    # Privacy 
    adjacency_matrix = np.array(W_ring(0) != 0, dtype=float)
    adjacency_matrix = adjacency_matrix - np.diag(np.diag(adjacency_matrix))
    degree_matrix = np.diag(adjacency_matrix @ np.ones_like(adjacency_matrix[0]))

    # eps_rdp_iteration = rdp_account(sigma, sigma_cor, c_clip, degree_matrix, adjacency_matrix, sparse=False, precision=0.1)
    eps_rdp_iteration = dp_account.rdp_account(sigma_cdp, sigma_cor, c_clip, degree_matrix, adjacency_matrix)

    # Learning
    sigma_ldp = c_clip * np.sqrt(2/eps_rdp_iteration)
    errors_centr, _ = optimizers.optimize_decentralized_correlated(X, W_centr, A, B, gamma, sigma_cdp, sigma_cor, c_clip, num_gossip=num_gossip, num_iter=num_iter)
    errors_cor, _ = optimizers.optimize_decentralized_correlated(X, W_ring, A, B, gamma, sigma_cdp, sigma_cor, c_clip, num_gossip=num_gossip, num_iter=num_iter)
    errors_ldp, _ = optimizers.optimize_decentralized_correlated(X, W_ring, A, B, gamma, sigma_ldp, 0, c_clip, num_gossip=num_gossip, num_iter=num_iter)

    fig, ax = plt.subplots()
    ax.semilogy(errors_centr, label="CDP")
    ax.semilogy(errors_cor, label="correlated DSGD")
    ax.semilogy(errors_ldp, label="LDP")
    ax.set_xlabel('iteration')
    ax.set_ylabel('loss')
    ax.set_title(f"loss with privacy eps per iteration {eps_rdp_iteration}")
    ax.legend()
    if target_eps:
        folder_path = f'./comparison_losses/epsilon = {target_eps}'
    else:
        folder_path = './comparison_losses'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plt.savefig('comparison_losses/loss-n_{}-d_{}-lr_{}-clip_{}-sigmacdp_{}-sigmacor_{}-sigmaldp_{}.png'.format(num_nodes, num_dim, gamma, c_clip, round(sigma_cdp, 2) , round(sigma_cor, 2), round(sigma_ldp, 2)))


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
    
    if abs(eps - eps_target) < 1e-4: # found
        return [sigma_cor]
    elif eps > eps_target: # increase sigma
        return find_sigma_cor(sigma_cdp, sigma_cor_grid[n // 2 :], c_clip, degree_matrix, adjacency_matrix, eps_target)
    else: #eps < eps_target
        return find_sigma_cor(sigma_cdp, sigma_cor_grid[:n // 2], c_clip, degree_matrix, adjacency_matrix, eps_target)


        
# Function to find the best couples (sigma_cdp, sigma_cor) in terms of loss for a fixed privacy 'target_eps'
def find_best_params(A, B, gamma, num_nodes, num_dim, max_loss, target_eps, c_clip, num_gossip=1, num_iter= 1000):
    """
    This function searches for the best parameters sigma_cdp and sigma_cor 
    such that they give a good performance (loss_cdp < min_loss) and a privacy under target_eps

    Args:
        A (array): parameter A
        B (array): parameter B
        gamma (float): learning rate
        num_nodes (int): number of nodes
        num_dim (int): number of parameters for each worker
        max_loss (float): maximum loss for CDp
        target_eps (float): maximum user-privacy
    
        Return:
            sigma_cdp, sigma_cor, eps, loss_cdp (dict)

    """
    X = np.ones(shape=(num_dim, num_nodes))
    W_ring = topology.FixedMixingMatrix("ring", num_nodes)
    W_centr = topology.FixedMixingMatrix("centralized", num_nodes)

    # Privacy 
    adjacency_matrix = np.array(W_ring(0) != 0, dtype=float)
    adjacency_matrix = adjacency_matrix - np.diag(np.diag(adjacency_matrix))
    degree_matrix = np.diag(adjacency_matrix @ np.ones_like(adjacency_matrix[0]))

    # sigma_ldp
    sigma_ldp = c_clip * np.sqrt(2/target_eps)
    sigma_cdp_grid = np.linspace(sigma_ldp/np.sqrt(num_nodes), sigma_ldp, 100)
    print(f"lower bound sigma cdp {sigma_ldp/np.sqrt(num_nodes)}")
    print(f"upper bound sigma cdp {sigma_ldp}")

    # Initialization
    eps_0 = dp_account.rdp_account(sigma_cdp_grid[0], 0, c_clip, degree_matrix, adjacency_matrix)
    errors_centr, _ = optimizers.optimize_decentralized_correlated(X, W_centr, A, B, gamma, sigma_cdp_grid[0], 0, c_clip, num_gossip=num_gossip, num_iter=num_iter)

    data = [{"sigma_cdp": sigma_ldp/np.sqrt(num_nodes), "sigma_cor": 0, "eps": eps_0, "loss_cdp":errors_centr[-1], "loss_cor": 10}]
    result = pd.DataFrame(data)

    # Searching
    for sigma_cdp in sigma_cdp_grid:
        print(f"loop for sigma_cdp {sigma_cdp}")
        # Looking for sigma_cor for which dp_account < target_eps
        sigma_cor_grid = np.linspace(1, 100, 1000)
        all_sigma_cor = find_sigma_cor(sigma_cdp, sigma_cor_grid, c_clip, degree_matrix, adjacency_matrix, target_eps)
        print(f"sigma_cor {all_sigma_cor}")
        # Now test on the loss condition
        if len(all_sigma_cor) != 0: # Not empty
            for sigma_cor in all_sigma_cor:
                errors_centr, _ = optimizers.optimize_decentralized_correlated(X, W_centr, A, B, gamma, sigma_cdp, sigma_cor, c_clip, num_gossip=num_gossip, num_iter=num_iter)
                errors_cor, _ = optimizers.optimize_decentralized_correlated(X, W_ring, A, B, gamma, sigma_cdp, sigma_cor, c_clip, num_gossip=num_gossip, num_iter=num_iter)

                if errors_centr[-1] <= max_loss and result["loss_cor"].iloc[-1] - np.mean(errors_cor[800:]) > 0:
                    new_row = {"sigma_cdp":sigma_cdp, 
                               "sigma_cor": sigma_cor, "eps": dp_account.rdp_account(sigma_cdp, sigma_cor, c_clip, degree_matrix, adjacency_matrix),
                                "loss_cdp": errors_centr[-1],
                                 "loss_cor": errors_cor[-1] }
                    result = pd.concat([result, pd.DataFrame([new_row])], ignore_index=True)
                    #result = result.append(new_row, ignore_index = True)
                    print(f"added with privacy {dp_account.rdp_account(sigma_cdp, sigma_cor, c_clip, degree_matrix, adjacency_matrix)}")

        else:
            continue
    return result


def run_optimize(**inparams):
    np.random.seed(inparams["seed"])
    n, d = inparams["num_nodes"], inparams["num_dim"]
    A, B = inparams["A"], inparams["B"]
    X = np.ones(shape=(d, n))
    topo = topology.FixedMixingMatrix(inparams["topology"], n)
    errors, _ = optimizers.optimize_decentralized_correlated(X, topo, A, B, inparams["lr"], inparams["sigma_cdp"],
                                                  inparams["sigma_cor"], inparams["clip"],
                                                  num_gossip=inparams["num_gossip"], num_iter=inparams["num_iter"])

    return errors

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


