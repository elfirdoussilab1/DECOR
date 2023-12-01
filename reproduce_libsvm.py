# coding: utf-8
###
 # @file   reproduce_mnist.py
 # @author Aymane El Firdoussi <aymane.elfirdoussi@epfl.ch>
 #
 # Running Correlated Decentralized learning experiments on LibSVM.
###

from utils import dp_account, plotting, topology
import tools, misc, study
tools.success("Module loading...")
import signal, torch
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------- #
# Miscellaneous initializations
tools.success("Miscellaneous initializations...")

# "Exit requested" global variable accessors
exit_is_requested, exit_set_requested = tools.onetime("exit")

# Signal handlers
signal.signal(signal.SIGINT, exit_set_requested)
signal.signal(signal.SIGTERM, exit_set_requested)

# ---------------------------------------------------------------------------- #

#JS: Pick the dataset on which to run experiments
dataset = "libsvm"
result_directory = "results-data-" + dataset 
plot_directory = "results-plot-" + dataset

with tools.Context("cmdline", "info"):
    args = misc.process_commandline()
    # Make the result directories
    args.result_directory = misc.check_make_dir(result_directory)
    args.plot_directory = misc.check_make_dir(plot_directory)
    # Preprocess/resolve the devices to use
    if args.devices == "auto":
        if torch.cuda.is_available():
            args.devices = list(f"cuda:{i}" for i in range(torch.cuda.device_count()))
        else:
            args.devices = ["cpu"]
    else:
        args.devices = list(name.strip() for name in args.devices.split(","))

# ---------------------------------------------------------------------------- #
# Run (missing) experiments
tools.success("Running experiments...")

# Base parameters for the MNIST experiments
params = {
    "dataset":  dataset,
    #"batch-size": 25,
    "batch-size": 64,
    "loss": "BCELoss",
    "learning-rate-decay-delta": 200,
    "learning-rate-decay": 200,
    "weight-decay": 1e-5,
    "evaluation-delta": 5,
    "gradient-clip": 0.1,
    "num-iter": 500,
    "num-nodes": 16,
    "momentum": 0.,
    "num-labels": 2,
    "delta": 1e-5,
    "criterion": "libsvm_criterion",
    "privacy": "user",
    "metric": "Loss"
    }

# Hyperparameters to test
models = ["libsvm_model"]
topologies = [("centralized", "cdp"), ("grid", "corr"), ("ring", "corr"), ("centralized", "ldp") , ("grid", "ldp"), ("ring", "ldp")]
alphas = [1.]
epsilons = [3, 5, 7, 10, 15, 20, 25, 30, 40]


# Command maker helper
def make_command(params):
    cmd = ["python3", "-OO", "train.py"]
    cmd += tools.dict_to_cmdlist(params)
    return tools.Command(cmd)

# Jobs
jobs  = tools.Jobs(args.result_directory, devices=args.devices, devmult=args.supercharge)
seeds = jobs.get_seeds()

# Submit all experiments
for alpha in alphas:
    for model in models:
        for target_eps in epsilons:
                for topology_name, method in topologies:
                    params["model"] = model
                    params["dirichlet-alpha"] = alpha
                    params["topology-name"] = topology_name
                    params["method"] = method
                    params["epsilon"] = target_eps

                    # Privacy
                    W = topology.FixedMixingMatrix(topology_name, params["num-nodes"])
                    adjacency_matrix = np.array(W(0) != 0, dtype=float)
                    adjacency_matrix = adjacency_matrix - np.diag(np.diag(adjacency_matrix))
                    degree_matrix = np.diag(adjacency_matrix @ np.ones_like(adjacency_matrix[0]))
                    eps_iter = dp_account.reverse_eps(eps= target_eps, num_iter = params["num-iter"], delta = params["delta"], subsample = 1, multiple = False)

                    # sigma_cdp and sigma_ldp
                    params["gradient-clip"] = 0.1
                    sigma_ldp = params["gradient-clip"] * np.sqrt(2 / eps_iter)
                    sigma_cdp = sigma_ldp / np.sqrt(params["num-nodes"])

                    if "corr" in method: # CD-SGD
                        params["learning-rate"] = 0.1 # To adapt with the result of the tuning
                        
                        # Determining the couples (sigma, sigma_cor) that can be considered
                        df = pd.DataFrame(columns = ["topology", "sigma", "sigma-cor", "epsilon", "sigma-cdp", "sigma-ldp"])
                        sigma_grid = np.linspace(sigma_cdp, sigma_ldp, 50)
                        sigma_cor_grid = np.linspace(1, 100, 1000)
                        for sigma in sigma_grid:
                            all_sigma_cor = plotting.find_sigma_cor(sigma, sigma_cor_grid, params["gradient-clip"], degree_matrix, adjacency_matrix, eps_iter)
                            # check non-emptyness and add it
                            if len(all_sigma_cor) !=0:
                
                                new_row = {"topology": topology_name,
                                           "sigma": sigma,
                                           "sigma-cor": all_sigma_cor[0],
                                           "epsilon": target_eps,
                                           "sigma-cdp": sigma_cdp,
                                           "sigma-ldp": sigma_ldp}
                                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

                        # TODO: adapt it with the result of the tuning (see which values go the best!)
                        # Taking the values on the first row (correspond to the least sigma): 
                        params["sigma"] = df.iloc[-1]["sigma"]
                        params["sigma-cor"] = df.iloc[-1]["sigma-cor"]
                        
                        # Store result
                        filename= f"result_gridsearch_user-level_{topology_name}_epsilon_{target_eps}.csv"
                        df.to_csv(filename)
                        jobs.submit(f"{dataset}-{topology_name}-{method}-n_{params['num-nodes']}-model_{model}-alpha_{alpha}-eps_{target_eps}", make_command(params))

                    elif "ldp" in method: # LDP
                        params["learning-rate"] = 0.1 # To adapt with the result of the tuning
                        params["sigma-cor"] = 0
                        params["sigma"] = sigma_ldp
                        #tools.success("Submitting LDP")
                        jobs.submit(f"{dataset}-{topology_name}-{method}-n_{params['num-nodes']}-model_{model}-alpha_{alpha}-eps_{target_eps}", make_command(params))
                    
                    else: # CDP
                        params["learning-rate"] = 0.1 # To adapt with the result of the tuning
                        params["sigma-cor"] = 0
                        params["sigma"] = sigma_cdp
                        #tools.success("Submitting CDP")
                        jobs.submit(f"{dataset}-{topology_name}-{method}-n_{params['num-nodes']}-model_{model}-alpha_{alpha}-eps_{target_eps}", make_command(params))

# Wait for the jobs to finish and close the pool
jobs.wait(exit_is_requested)
jobs.close()

# Check if exit requested before going to plotting the results
if exit_is_requested():
    exit(0)

# ---------------------------------------------------------------------------- #
# Plot results
tools.success("Plotting results...")

# dictionary for plot colors ans style
topo_to_style = {"ring": (0, (1, 1)), "grid": (0, (5, 5)), "centralized": 'solid'}
method_to_color = {"ldp": "tab:orange", "cdp": "tab:purple", "corr": "tab:green"}
method_to_marker = {"ldp": "^", "cdp": "s", "corr": "o"}

# Plot Loss VS iterations
# TODO: change plot.finalize to make the desired legend 
with tools.Context("mnist", "info"):
    for alpha in alphas:
        for model in models:
            for target_eps in epsilons:
                    values = dict()
                    plot = study.LinePlot()
                    legend = []
                    
                    for topology_name, method in topologies:
                        name = f"{dataset}-{topology_name}-{method}-n_{params['num-nodes']}-model_{model}-alpha_{alpha}-eps_{target_eps}"
                        values[topology_name, method] = misc.compute_avg_err_op(name, seeds, result_directory, "eval", (params["metric"], "max"))
                        plot.include(values[topology_name, method][0], params["metric"], errs="-err", linestyle = topo_to_style[topology_name], 
                        color = method_to_color[method], lalp=0.8, logscale = True)
                        legend.append(f"{topology_name} + {method}")

                    #JS: plot every time graph in terms of the maximum number of steps
                    plot_name = f"{dataset}_model= {model}_momentum={params['momentum']}_alpha={alpha}_eps={target_eps}"
                    plot.finalize(None, "Step number", "Test accuracy", xmin=0, xmax=params['num-iter'], ymin=1e-4, ymax=1, legend=legend)
                    plot.save(plot_directory + "/" + plot_name + ".pdf", xsize=3, ysize=1.5)

# Plot Loss VS Epsilon
# TODO: change the order of for loops 
with tools.Context("mnist", "info"):
    for alpha in alphas:
        for model in models:
            for target_eps in epsilons:
                    values = dict()
                    plot = study.LinePlot()
                    for topology_name, method in topologies:
                        name = f"{dataset}-{topology_name}-{method}-n_{params['num-nodes']}-model_{model}-alpha_{alpha}-eps_{target_eps}"
                        values[topology_name, method] = misc.compute_avg_err_op(name, seeds, result_directory, "eval", (params["metric"], "max"))
                        plot.include(values[topology_name, method][0], params["metric"], errs="-err", linestyle = topo_to_style[topology_name], 
                                     marker = method_to_marker[method], color = method_to_color[method], lalp=0.8, logscale = True)

                    #JS: plot every time graph in terms of the maximum number of steps
                    plot_name = f"{dataset}_model= {model}_momentum={params['momentum']}_alpha={alpha}_eps={target_eps}"
                    plot.finalize(None, "Step number", "Test accuracy", xmin=0, xmax=params['num-iter'], ymin=1e-4, ymax=1)
                    plot.save(plot_directory + "/" + plot_name + ".pdf", xsize=3, ysize=1.5)
