# This file is used to tune our CDP model
import torch
from models import *
import misc, os, worker, dataset, evaluator
from utils import dp_account, topology, plotting
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Fix parameters
model = "libsvm_model"
dataset_name = "libsvm"
loss = "BCELoss"
num_nodes = 16
num_labels = 2
alpha = 1.
delta = 1e-5
target_eps = 10
min_loss = 0.3236 # found in train_libsvm_bce.ipynb
criterion = "libsvm_topk"

# Hyper-parameters
lr_grid = [0.01, 0.05, 0.1, 0.5]
gradient_clip_grid = [0.01, 0.05, 0.1, 0.5, 1]
T_grid = [500, 1000, 2000]
batch_size = 64
momentum = 0.
weight_decay = 1e-5
topology_name = "ring"
method = "corr"

# Fix seed
misc.fix_seed(1)

# Storing reults
evaluation_delta = 5

result_directory = "./results-tuning-" + dataset_name + "-" + method  + "-" + topology_name
if not os.path.exists(result_directory):
    os.makedirs(result_directory)

# Create train and test dataloaders
train_loader_dict, test_loader = dataset.make_train_test_datasets(dataset=dataset_name, num_labels=num_labels, 
                                alpha_dirichlet= alpha, num_nodes=num_nodes, train_batch=batch_size, test_batch=100)

def train_decentralized(topology_name, method, sigma, sigma_cor, lr, gradient_clip, min_loss, num_iter):
    
    # Testing model
    server = evaluator.Evaluator(train_loader_dict, test_loader, model, loss, num_labels, criterion, num_evaluations= 100, device=device)

    # Initialize Workers
    workers = []
    for i in range(num_nodes):
        data_loader = train_loader_dict[i]
        worker_i = worker.Worker(train_data_loader=data_loader, test_data_loader=test_loader, batch_size=batch_size, 
                    model = model, loss = loss, momentum = momentum, gradient_clip= gradient_clip, sigma= sigma,
                    num_labels= num_labels, criterion= criterion, num_evaluations= 100, device= device, privacy = "user")
        # Agree on first parameters
        worker_i.flat_parameters = server.flat_parameters
        worker_i.update_model_parameters()
        workers.append(worker_i)
    
    # Noise tensor: shape (num_nodes, num_nodes, model_size)
    V = torch.randn(num_nodes, num_nodes, workers[0].model_size) # distribution N(0, 1)
    V.mul_(sigma_cor) # rescaling ==> distribution N (0, sigma_cor^2)

    # Antisymmetry property
    V = misc.to_antisymmetric(V).to(device)

    # ------------------------------------------------------------------------ #
    current_step = 0
    eval_filename = result_directory + f"/mean_loss-{dataset_name}-{topology_name}-{method}-lr-{lr}-clip-{gradient_clip}-sigma-{sigma}-sigmacor-{sigma_cor}-epsilon-{target_eps}-T-{num_iter}.csv"
    # Initialization of the dataframe
    data = [{"Step": -1, "topology": topology_name, "method": method, "lr": lr, "clip" : gradient_clip, "sigma": sigma, "sigma_cor": sigma_cor, 
                "epsilon": target_eps, "loss":-1}]
    result = pd.DataFrame(data)
    
    # Training
    while current_step <= num_iter:
        
        # Evaluate the model if milestone is reached
        milestone_evaluation = evaluation_delta > 0 and current_step % evaluation_delta == 0        
        if milestone_evaluation:
            mean_param = torch.stack([workers[i].flat_parameters]).mean(dim = 0)
            server.update_model_parameters(mean_param)
            mean_loss = server.compute_train_loss() - min_loss
            new_row = {"Step": current_step,
                        "topology": topology_name,
                        "method": method,
                        "lr": lr, 
                        "clip": gradient_clip,
                        "sigma": sigma,
                        "sigma_cor": sigma_cor,
                        "epsilon": target_eps,
                        "loss": mean_loss                
                        }
            result = pd.concat([result, pd.DataFrame([new_row])], ignore_index=True)
            result.set_index(['Step']).to_csv(eval_filename)
        # Apply the algorithm
        all_parameters = []

        # Step t + 1/2
        for i in range(num_nodes):
            all_parameters.append(workers[i].grad_descent(V[i], lr = lr, weight_decay = weight_decay))
        
        all_parameters = torch.stack(all_parameters).to(device)
        
        # Step t + 1
        for i in range(num_nodes):
            workers[i].decentralized_learning(weights = W[i], workers_parameters = all_parameters)

        current_step += 1
    return result.iloc[-1]["loss"]

# Creating a dictionary that will contain the values of loss for all couples considered, and will be sorted
summary = pd.DataFrame(columns = ["topology", "lr", "clip", "sigma", "sigma-cor", "T", "loss"])

# Tuning: looping over the hyperparameters
for lr in lr_grid:
    for gradient_clip in gradient_clip_grid:
        for num_iter in T_grid:
            # Weights matrix
            W = topology.FixedMixingMatrix(topology_name= topology_name, n_nodes= num_nodes)(0)
            adjacency_matrix = np.array(W != 0, dtype=float)
            adjacency_matrix = adjacency_matrix - np.diag(np.diag(adjacency_matrix))
            degree_matrix = np.diag(adjacency_matrix @ np.ones_like(adjacency_matrix[0]))
            # Convert it to tensor
            W = torch.tensor(W, dtype= torch.float).to(device)
            print(W[0])

            # Determining noise for CDP
            eps_iter = dp_account.reverse_eps(target_eps, num_iter, delta, subsample = 1.)
            sigma_ldp = gradient_clip * np.sqrt(2 / eps_iter)
            sigma_cdp =  sigma_ldp / np.sqrt(num_nodes)
            sigma_cor = 0

            if "cdp" in method:
                sigma = sigma_cdp
                final_loss= train_decentralized(topology_name, method, sigma, sigma_cor, lr, gradient_clip, min_loss, num_iter)
                row = {"topology": topology_name,
                    "lr": lr,
                    "clip": gradient_clip,
                    "sigma": sigma,
                    "sigma-cor": sigma_cor,
                    "T": num_iter,
                    "loss": final_loss}
                summary = pd.concat([summary, pd.DataFrame([row])], ignore_index=True)

            elif "ldp" in method:
                sigma = sigma_ldp
                final_loss = train_decentralized(topology_name, method, sigma, sigma_cor, lr, gradient_clip, min_loss, num_iter)
                row = {"topology": topology_name,
                    "lr": lr,
                    "clip": gradient_clip,
                    "sigma": sigma,
                    "sigma-cor": sigma_cor,
                    "T": num_iter,
                    "loss": final_loss}
                summary = pd.concat([summary, pd.DataFrame([row])], ignore_index=True)
            else: # corr
                # Determining the couples (sigma, sigma_cor) that can be considered
                sigmas_df = pd.DataFrame(columns = ["topology", "sigma", "sigma-cor", "epsilon"])
                sigma_grid = np.linspace(sigma_cdp, sigma_ldp, 50)
                sigma_cor_grid = np.linspace(1e-4, 100, 1000)

                for sigma in sigma_grid:
                    all_sigma_cor = plotting.find_sigma_cor(sigma, sigma_cor_grid, gradient_clip, degree_matrix, adjacency_matrix, eps_iter)
                    # check non-emptyness and add it
                    if len(all_sigma_cor) !=0:
                        eps = dp_account.rdp_compose_convert(num_iter, delta, sigma, all_sigma_cor[0], gradient_clip,
                                                                              degree_matrix, adjacency_matrix, subsample =1)
                        new_row = {"topology": topology_name,
                                    "sigma": sigma,
                                    "sigma-cor": all_sigma_cor[0],
                                    "epsilon": eps}
                        if int(eps) == target_eps:
                            sigmas_df = pd.concat([sigmas_df, pd.DataFrame([new_row])], ignore_index=True)

                # Store result of ooking for sigmas
                filename = f'result_gridsearch_tuning_{topology_name}_corr_{target_eps}_clip_{gradient_clip}_T_{num_iter}.csv'
                sigmas_df.to_csv(filename)

                # Taking the values on the first row (correspond to the least sigma)
                # Selecting values of (sigma, sogma_cor) based on the length of the result
                n = sigmas_df.shape[0]
                if n <= 0:
                    continue
                elif n == 1:
                    # single couple
                    sigma = sigmas_df.iloc[0]["sigma"]
                    sigma_cor = sigmas_df.iloc[0]["sigma-cor"]
                    final_loss = train_decentralized(topology_name, method, sigma, sigma_cor, lr, gradient_clip, min_loss, num_iter)
                    row = {"topology": topology_name,
                        "lr": lr,
                        "clip": gradient_clip,
                        "sigma": sigma,
                        "sigma-cor": sigma_cor,
                        "T": num_iter,
                        "loss": final_loss}
                    summary = pd.concat([summary, pd.DataFrame([row])], ignore_index=True)
                elif n == 2:
                    # First couple
                    sigma = sigmas_df.iloc[0]["sigma"]
                    sigma_cor = sigmas_df.iloc[0]["sigma-cor"]
                    final_loss = train_decentralized(topology_name, method, sigma, sigma_cor, lr, gradient_clip, min_loss, num_iter)
                    row = {"topology": topology_name,
                        "lr": lr,
                        "clip": gradient_clip,
                        "sigma": sigma,
                        "sigma-cor": sigma_cor,
                        "T": num_iter,
                        "loss": final_loss}
                    summary = pd.concat([summary, pd.DataFrame([row])], ignore_index=True)

                    # Second couple
                    sigma = sigmas_df.iloc[1]["sigma"]
                    sigma_cor = sigmas_df.iloc[1]["sigma-cor"]
                    final_loss = train_decentralized(topology_name, method, sigma, sigma_cor, lr, gradient_clip, min_loss, num_iter)
                    row = {"topology": topology_name,
                        "lr": lr,
                        "clip": gradient_clip,
                        "sigma": sigma,
                        "sigma-cor": sigma_cor,
                        "T": num_iter,
                        "loss": final_loss}
                    summary = pd.concat([summary, pd.DataFrame([row])], ignore_index=True)
                else: #n > 3
                    # First couple
                    sigma = sigmas_df.iloc[0]["sigma"]
                    sigma_cor = sigmas_df.iloc[0]["sigma-cor"]
                    final_loss = train_decentralized(topology_name, method, sigma, sigma_cor, lr, gradient_clip, min_loss, num_iter)
                    row = {"topology": topology_name,
                        "lr": lr,
                        "clip": gradient_clip,
                        "sigma": sigma,
                        "sigma-cor": sigma_cor,
                        "T": num_iter,
                        "loss": final_loss}
                    summary = pd.concat([summary, pd.DataFrame([row])], ignore_index=True)
                    # Second couple
                    sigma = sigmas_df.iloc[n//2]["sigma"]
                    sigma_cor = sigmas_df.iloc[n//2]["sigma-cor"]
                    final_loss = train_decentralized(topology_name, method, sigma, sigma_cor, lr, gradient_clip, min_loss, num_iter)
                    row = {"topology": topology_name,
                        "lr": lr,
                        "clip": gradient_clip,
                        "sigma": sigma,
                        "sigma-cor": sigma_cor,
                        "T": num_iter,
                        "loss": final_loss}
                    summary = pd.concat([summary, pd.DataFrame([row])], ignore_index=True)
                    # Last couple
                    sigma = sigmas_df.iloc[-1]["sigma"]
                    sigma_cor = sigmas_df.iloc[-1]["sigma-cor"]
                    final_loss = train_decentralized(topology_name, method, sigma, sigma_cor, lr, gradient_clip, min_loss, num_iter)
                    row = {"topology": topology_name,
                        "lr": lr,
                        "clip": gradient_clip,
                        "sigma": sigma,
                        "sigma-cor": sigma_cor,
                        "T": num_iter,
                        "loss": final_loss}
                    summary = pd.concat([summary, pd.DataFrame([row])], ignore_index=True)

                summary.to_csv(result_directory + f"/summary-tuning-libsvm-{topology_name}-{method}-epsilon-{target_eps}.csv")


# Produce the last file
sorted_summary = summary.sort_values(by='loss')
sorted_summary.to_csv(result_directory + f"/sorted-summary-tuning-libsvm-{topology_name}-{method}-epsilon-{target_eps}.csv")

'''
for lr in lr_grid:
    for gradient_clip in gradient_clip_grid:
        for num_iter in T_grid:
            filename = result_directory + f"/mean_loss-{dataset_name}-{topology_name}-{method}-lr-{lr}-clip-{gradient_clip}-epsilon-{target_eps}-T-{num_iter}.csv"
            df = pd.read_csv(filename)
            row = {"topology": topology_name,
                    "lr": lr,
                    "clip": gradient_clip,
                    "sigma": df.iloc[-1]["sigma"],
                    "sigma_cor": df.iloc[-1]["sigma_cor"],
                    "T": num_iter,
                    "loss": df.iloc[-1]["loss"]}
            summary = pd.concat([summary, pd.DataFrame([row])], ignore_index=True)

'''
