# coding: utf-8
###
 # @file   reproduce_mnist.py
 #
 # Running Correlated Decentralized learning experiments on MNIST.
###

from utils import dp_account, plotting, topology
import tools, misc, study
tools.success("Module loading...")
import signal, torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
dataset = "mnist"
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
    "dataset": "mnist",
    "batch-size": 64,
    "loss": "CrossEntropyLoss",
    "weight-decay": 1e-5,
    "evaluation-delta": 5,
    "num-iter": 1000,
    "num-nodes": 16,
    "momentum": 0.,
    "num-labels": 10,
    "delta": 1e-5, 
    "privacy": "example",
    "metric": "Accuracy",
    "hetero": False,
    "gradient-descent": False
    }

# Hyperparameters to test
models = ["simple_mnist_model"]
topologies = [("centralized", "cdp"), ("centralized", "corr"), ("grid", "corr"), ("ring", "corr"), ("centralized", "ldp") , ("grid", "ldp"), ("ring", "ldp")]
alphas = [10]
epsilons = [1e-3, 1e-2, 0.1, 1]
tick_labels = ['$10^{-3}$', '$10^{-2}$', '$10^{-1}$', 1 ]

hyperparam_dict = {("centralized", "cdp", 0.1) : (5, 1), ("centralized", "cdp", 0.5): (5, 1), ("centralized", "cdp", 1) : (5, 1), ("centralized", "cdp", 3): (5, 1),
                   ("centralized", "ldp", 0.1) : (1, 1), ("centralized", "ldp", 0.5): (5, 1), ("centralized", "ldp", 1) : (1, 1), ("centralized", "ldp", 3): (1, 1), 
                   ("grid", "ldp", 0.1) : (1, 1), ("grid", "ldp", 0.5): (5, 1), ("grid", "ldp", 1) : (0.5, 1), ("grid", "ldp", 3): (1, 1), 
                   ("ring", "ldp", 0.1) : (1, 1), ("ring", "ldp", 0.5): (5, 1), ("ring", "ldp", 1) : (1, 1), ("ring", "ldp", 3): (1, 1), 
                   ("centralized", "corr", 0.1) : (5, 1), ("centralized", "corr", 0.5): (5, 1), ("centralized", "corr", 1) : (5, 1), ("centralized", "corr", 3): (5, 1),
                   ("grid", "corr", 0.1) : (5, 1), ("grid", "corr", 0.5): (5, 1), ("grid", "corr", 1) : (1, 5), ("grid", "corr", 3): (1, 5), 
                   ("ring", "corr", 0.1) : (2, 1), ("ring", "corr", 0.5): (1, 1.5), ("ring", "corr", 1) : (1, 1.5), ("ring", "corr", 3): (1, 1.5)
}

# Command maker helper
def make_command(params):
    cmd = ["python3", "-OO", "train.py"]
    cmd += tools.dict_to_cmdlist(params)
    return tools.Command(cmd)

# Jobs
jobs  = tools.Jobs(args.result_directory, devices=args.devices, devmult=args.supercharge)
seeds = jobs.get_seeds()

# Dataset to total number of samples
dataset_samples = {"mnist": 60000}

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

                # hyperparams
                if target_eps < 0.1:
                    params["learning-rate"], params["gradient-clip"] = (1, 1)
                else:
                    params["learning-rate"], params["gradient-clip"] = hyperparam_dict[topology_name, method, target_eps]
                # Training model without noise
                #jobs.submit(f"{dataset}-average-n_{params['num-nodes']}-model_{model}-lr_{lr}-momentum_{params['momentum']}-alpha_{alpha}", make_command(params))

                # Privacy
                W = topology.FixedMixingMatrix(topology_name, params["num-nodes"])
                adjacency_matrix = np.array(W(0) != 0, dtype=float)
                adjacency_matrix = adjacency_matrix - np.diag(np.diag(adjacency_matrix))
                degree_matrix = np.diag(adjacency_matrix @ np.ones_like(adjacency_matrix[0]))

                subsample = params["batch-size"] / (dataset_samples[params["dataset"]] / params["num-nodes"])
                eps_iter = dp_account.reverse_eps(target_eps, params["num-iter"], params["delta"], params["num-nodes"], params["gradient-clip"], 
                                                    topology_name, degree_matrix, adjacency_matrix, subsample, params["batch-size"], multiple = True)

                # sigma_cdp and sigma_ldp
                sigma_ldp = params["gradient-clip"] * np.sqrt(2 / eps_iter)
                sigma_cdp = sigma_ldp / np.sqrt(params["num-nodes"])

                if "corr" in method: # CD-SGD
                    # Determining the couples (sigma, sigma_cor) that can be considered
                    filename= f"result_gridsearch_example-level_{topology_name}_epsilon_{target_eps}.csv"
                    df = pd.read_csv(filename)
                
                    # Taking the values on the first row (correspond to the least sigma)
                    params["sigma"] =  df.iloc[0]["sigma"]
                    params["sigma-cor"] = df.iloc[0]["sigma-cor"]
                    jobs.submit(f"{dataset}-{topology_name}-{method}-n_{params['num-nodes']}-model_{model}-alpha_{alpha}-eps_{target_eps}", make_command(params))

                elif "ldp" in method: # LDP
                    params["sigma-cor"] = 0
                    params["sigma"] = sigma_ldp
                    #tools.success("Submitting LDP")
                    jobs.submit(f"{dataset}-{topology_name}-{method}-n_{params['num-nodes']}-model_{model}-alpha_{alpha}-eps_{target_eps}", make_command(params))
                
                else: # CDP
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
method_to_marker = {"ldp": "^", "cdp": "D", "corr": "o"}
method_to_legend = {"ldp": "LDP", "corr": "DECOR", "cdp": "CDP"}

# Plot Loss VS iterations
with tools.Context("mnist", "info"):
    for alpha in alphas:
        for model in models:
            for target_eps in epsilons:
                values = dict()
                plot = study.LinePlot()
                legend_topos = []
                legend_methods = []
                
                for topology_name, method in topologies:
                    name = f"{dataset}-{topology_name}-{method}-n_{params['num-nodes']}-model_{model}-alpha_{alpha}-eps_{target_eps}"
                    values[topology_name, method] = misc.compute_avg_err_op(name, seeds, result_directory, "eval", (params["metric"], "max"))
                    plot.include(values[topology_name, method][0], params["metric"], errs="-err", linestyle = topo_to_style[topology_name], 
                    color = method_to_color[method], lalp=0.8)
                    #legend.append(f"{topology_name} + {method}")
                    if topology_name not in legend_topos:
                        legend_topos.append(topology_name)
                    if method not in legend_methods:
                        legend_methods.append(method)
                
                # Making the legend
                legend = []
                legend.append(plt.Line2D([], [], label='Algorithm', linestyle = 'None'))
                for method in legend_methods:
                    legend.append(plt.Line2D([], [], label=method_to_legend[method], color = method_to_color[method]))
                legend.append(plt.Line2D([], [], label='Topology', linestyle = 'None'))
                for topo in legend_topos:
                    legend.append(plt.Line2D([], [], label= topo.capitalize(), linestyle = topo_to_style[topo], color = 'k'))
                    
                #JS: plot every time graph in terms of the maximum number of steps
                plot_name = f"{dataset}_model= {model}_momentum={params['momentum']}_alpha={alpha}_eps={target_eps}"
                plot.finalize(None, "Step number", "Test Accuracy", xmin=0, xmax=params['num-iter'], ymin = 0.8, ymax= 1, legend=legend)
                plot.save(plot_directory + "/" + plot_name + ".pdf", xsize=3, ysize=1.5)

# Plot Loss VS Epsilon
# Checked !

with tools.Context("libsvm", "info"):
    for alpha in alphas:
        for model in models:
            plot = study.LinePlot()
            #legend_topos = []
            #legend_methods = []
            for topology_name, method in topologies:
                #if topology_name not in legend_topos:
                #    legend_topos.append(topology_name)
                #if method not in legend_methods:
                #    legend_methods.append(method)

                values = pd.DataFrame(columns = ["epsilon", params["metric"], params["metric"] +"-err"])
                for target_eps in epsilons:
                    name = f"{dataset}-{topology_name}-{method}-n_{params['num-nodes']}-model_{model}-alpha_{alpha}-eps_{target_eps}"
                    df = misc.compute_avg_err_op(name, seeds, result_directory, "eval", (params["metric"], "max"))[0]
                    new_row = {"epsilon": target_eps,
                                params["metric"]: df.iloc[-1][params["metric"]],
                                params["metric"] +"-err" : df.iloc[-1][params["metric"] +"-err"]}
                    values = pd.concat([values, pd.DataFrame([new_row])], ignore_index=True)
                    

                plot.include(values, params["metric"], errs="-err", xticks = epsilons, linestyle = topo_to_style[topology_name], 
                                    mark = method_to_marker[method], color = method_to_color[method], lalp=0.8, xlogscale= True)
            # Making the legend
            #legend = []
            #legend.append(plt.Line2D([], [], label='Algorithm', linestyle = 'None' ))
            #for method in legend_methods:
            #    legend.append(plt.Line2D([], [], label=method_to_legend[method], color = method_to_color[method], marker = method_to_marker[method]))
            #legend.append(plt.Line2D([], [], label='Topology', linestyle = 'None'))
            #for topo in legend_topos:
            #    legend.append(plt.Line2D([], [], label= topo.capitalize(), linestyle = topo_to_style[topo], color = 'k'))

            #JS: plot every time graph in terms of the maximum number of steps
            plot_name = f"Accuracy_vs_epsilon_{dataset}_model={model}_momentum={params['momentum']}_alpha={alpha}"
            plot.finalize(title = None, xlabel="Example-level $\epsilon$", ylabel="Accuracy", tick_labels= tick_labels, xticks= epsilons)#, legend = legend)
            plot.save(plot_directory + "/" + plot_name + ".pdf", xsize=2, ysize=1.5)

