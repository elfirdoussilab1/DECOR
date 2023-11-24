# This file is used to tune our CDP model
import torch
from models import *
import misc, os, worker, dataset
from utils import dp_account, topology, plotting
import numpy as np
import pandas as pd


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Fix parameters
model = "simple_mnist_model"
dataset_name = "mnist"
batch_size_test = 100
loss = "NLLLoss"
num_nodes = 16
num_labels = 10
alpha = 1.
delta = 1e-5
target_eps = 30
num_iter = 500
criterion = "topk"
num_evaluations = 100

# Hyper-parameters
lr_grid = [0.01, 0.05, 0.1, 0.5, 1]
gradient_clip_grid = [2.]
batch_size = 64
momentum = 0.
weight_decay = 1e-4
topology_name = "centralized"
method = "cdp"

# Fix seed
misc.fix_seed(1)

# Storing reults
evaluation_delta = 5

result_directory = "./results-tuning-" + dataset_name
if not os.path.exists(result_directory):
    os.makedirs(result_directory)

# Create train and test dataloaders
train_loader_dict, test_loader = dataset.make_train_test_datasets(dataset=dataset_name, num_labels=num_labels, 
                                alpha_dirichlet= alpha, num_nodes=num_nodes, train_batch=batch_size, test_batch=batch_size_test)

# Tuning: looping over the hyperparameters
for lr in lr_grid:
    for gradient_clip in gradient_clip_grid:
        
        # Weights matrix
        W = topology.FixedMixingMatrix(topology_name= topology_name, n_nodes= num_nodes)(0)

        # Convert it to tensor
        W = torch.tensor(W, dtype= torch.float).to(device)
        print(W[0])

        # Determining noise for CDP
        eps_iter = dp_account.reverse_eps(target_eps, num_iter, delta)
        sigma_ldp = gradient_clip * np.sqrt(2 / eps_iter)
        sigma_cdp =  sigma_ldp / np.sqrt(num_nodes)
        sigma_cor = 0

        if "cdp" in method:
            sigma = sigma_cdp
        elif "ldp" in method:
            sigma = sigma_ldp
        else: # corr
            adjacency_matrix = np.array(W(0) != 0, dtype=float)
            adjacency_matrix = adjacency_matrix - np.diag(np.diag(adjacency_matrix))
            degree_matrix = np.diag(adjacency_matrix @ np.ones_like(adjacency_matrix[0]))

            # Determining the couples (sigma, sigma_cor) that can be considered
            sigmas_df = pd.DataFrame(columns = ["topology", "sigma", "sigma-cor", "epsilon"])
            sigma_grid = np.linspace(sigma_cdp, sigma_ldp, 50)
            sigma_cor_grid = np.linspace(1, 1000, 1000)

            for sigma in sigma_grid:
                all_sigma_cor = plotting.find_sigma_cor(sigma, sigma_cor_grid, gradient_clip, degree_matrix, adjacency_matrix, eps_iter)
                # check non-emptyness and add it
                if len(all_sigma_cor) !=0:

                    new_row = {"topology": topology_name,
                                "sigma": sigma,
                                "sigma-cor": all_sigma_cor[0],
                                "epsilon": target_eps}
                    sigmas_df = pd.concat([sigmas_df, pd.DataFrame([new_row])], ignore_index=True)

            # Store result of ooking for sigmas
            filename = f'result_gridsearch_{topology}_corr_{target_eps}.csv'
            sigmas_df.to_csv(filename)

            # Taking the values on the first row (correspond to the least sigma)
            sigma = sigmas_df.iloc[0]["sigma"]
            sigma_cor = sigmas_df.iloc[0]["sigma-cor"]

        
        # Initialize Workers
        workers = []
        for i in range(num_nodes):
            data_loader = train_loader_dict[i]
            worker_i = worker.Worker(train_data_loader=data_loader, test_data_loader=test_loader, batch_size=batch_size, 
                        model = model, loss = loss, momentum = momentum, gradient_clip= gradient_clip, sigma= sigma,
                        num_labels= num_labels, criterion= criterion, num_evaluations= num_evaluations, device= device)
            workers.append(worker_i)
        
        # Noise tensor: shape (num_nodes, num_nodes, model_size)
        V = torch.randn(num_nodes, num_nodes, workers[0].model_size) # distribution N(0, 1)
        V.mul_(sigma_cor) # rescaling ==> distribution N (0, sigma_cor^2)

        # Antisymmetry property
        V = misc.to_antisymmetric(V).to(device)

        # ------------------------------------------------------------------------ #

        current_step = 0
        eval_filename = result_directory + f"/mean_accuracy-{dataset_name}-lr-{lr}-clip-{gradient_clip}-epsilon-{target_eps}-T-{num_iter}.csv"
        # Initialization of the dataframe
        data = [{"Step": -1, "topology": topology_name, "method": method, "lr": lr, "clip" : gradient_clip, "sigma": sigma, "sigma_cor": sigma_cor, 
                    "epsilon": target_eps, "accuracy":0}]
        result = pd.DataFrame(data)
        
        # Training
        while current_step <= num_iter:
            
            # Evaluate the model if milestone is reached
            milestone_evaluation = evaluation_delta > 0 and current_step % evaluation_delta == 0        
            if milestone_evaluation:
                mean_accuracy = np.mean([workers[i].compute_accuracy() for i in range(num_nodes)])
                new_row = {"Step": current_step,
                            "topology": topology_name,
                            "method": method,
                            "lr": lr, 
                            "clip": gradient_clip,
                            "sigma": sigma,
                            "sigma_cor": sigma_cor,
                            "epsilon": target_eps,
                            "accuracy": mean_accuracy                
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