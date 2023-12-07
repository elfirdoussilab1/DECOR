# This file is used to tune our CDP model
import torch
from models import *
import misc, os, worker, dataset, evaluator
from utils import dp_account, topology, plotting
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Fix parameters
model = "simple_mnist_model"
dataset_name = "mnist"
batch_size_test = 100
loss = "CrossEntropyLoss"
num_nodes = 16
num_labels = 10 
alpha = 1000 # to have that each worker has approximatly 3750 samples
delta = 1e-5
target_eps = 0.5
criterion = "topk"
num_evaluations = 100

# Hyper-parameters
lr_grid = [0.005, 0.01, 0.05, 0.1, 0.5, 1]
gradient_clip_grid = [0.05, 0.1, 1., 1.5, 2., 4.]
num_iter = 5000
batch_size = 64
subsample = 64/3750
momentum = 0.
weight_decay = 1e-5
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

def train_decentralized(topology_name, method, sigma, sigma_cor, lr, gradient_clip, num_iter):
    # Testing model
    server = evaluator.Evaluator(train_loader_dict, test_loader, model, loss, num_labels, criterion, num_evaluations= num_evaluations, device=device)

    # Initialize Workers
    workers = []
    for i in range(num_nodes):
        data_loader = train_loader_dict[i]
        worker_i = worker.Worker(train_data_loader=data_loader, test_data_loader=test_loader, batch_size=batch_size, 
                    model = model, loss = loss, momentum = momentum, gradient_clip= gradient_clip, sigma= sigma,
                    num_labels= num_labels, criterion= criterion, num_evaluations= 100, device= device, privacy = "example")
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
    eval_filename = result_directory + f"/mean_accuracy-{dataset_name}-{topology_name}-{method}-lr-{lr}-clip-{gradient_clip}-mom-{momentum}-sigma-{sigma}-sigmacor-{sigma_cor}-epsilon-{target_eps}-T-{num_iter}.csv"
    plot_filename = result_directory + f"/mean_accuracy-{dataset_name}-{topology_name}-{method}-lr-{lr}-clip-{gradient_clip}-mom-{momentum}-sigma-{sigma}-sigmacor-{sigma_cor}-epsilon-{target_eps}-T-{num_iter}.png"    # Initialization of the dataframe
    result = pd.DataFrame(columns = ["Step", "topology", "method", "lr", "clip", "momentum", "sigma", "sigma-cor", "epsilon", "accuracy"])
    
    # Training
    while current_step <= num_iter:
        
        # Evaluate the model if milestone is reached
        milestone_evaluation = evaluation_delta > 0 and current_step % evaluation_delta == 0        
        if milestone_evaluation:
            #mean_accuracy = np.mean([workers[i].compute_accuracy() for i in range(num_nodes)])
            mean_param = torch.stack([workers[i].flat_parameters for i in range(num_nodes)]).mean(dim = 0)
            server.update_model_parameters(mean_param)
            mean_accuracy = server.compute_accuracy()
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
    fig, ax = plt.subplots()
    ax.semilogy(result["accuracy"], label = topology_name + method)
    ax.legend()
    fig.savefig(plot_filename)
    return np.mean(result.iloc[-200:-1]["accuracy"])

# Creating a dictionary that will contain the values of loss for all couples considered, and will be sorted
summary = pd.DataFrame(columns = ["topology", "lr", "clip", "momentum", "sigma", "sigma-cor", "T", "accuracy"])

# Tuning: looping over the hyperparameters
for lr in lr_grid:
    for gradient_clip in gradient_clip_grid:
        
        # Weights matrix
        W = topology.FixedMixingMatrix(topology_name= topology_name, n_nodes= num_nodes)(0)
        adjacency_matrix = np.array(W != 0, dtype=float)
        adjacency_matrix = adjacency_matrix - np.diag(np.diag(adjacency_matrix))
        degree_matrix = np.diag(adjacency_matrix @ np.ones_like(adjacency_matrix[0]))


        # Convert it to tensor
        W = torch.tensor(W, dtype= torch.float).to(device)
        print(W[0])

        # Determining eps iter
        eps_iter = dp_account.reverse_eps(target_eps, num_iter, delta, num_nodes, gradient_clip, 
                                            topology_name, degree_matrix, adjacency_matrix, subsample, batch_size, multiple = True)

        sigma_ldp = gradient_clip * np.sqrt(2 / eps_iter)
        sigma_cdp =  sigma_ldp / np.sqrt(num_nodes)
        sigma_cor = 0

        if "cdp" in method:
            sigma = sigma_cdp
            train_decentralized(topology_name, method, sigma, sigma_cor, lr, gradient_clip, num_iter)
        elif "ldp" in method:
            sigma = sigma_ldp
            train_decentralized(topology_name, method, sigma, sigma_cor, lr, gradient_clip, num_iter)
        else: # corr
            # Store result of looking for sigmas
            filename= f"result_gridsearch_example-level_{topology_name}_epsilon_{target_eps}.csv"
            df = pd.read_csv(filename)

            # Taking the values on the last row (correspond to the highest sigma)
            sigma = df.iloc[-1]["sigma"]
            sigma_cor = df.iloc[-1]["sigma-cor"]
            train_decentralized(topology_name, method, sigma, sigma_cor, lr, gradient_clip, num_iter)

        

        