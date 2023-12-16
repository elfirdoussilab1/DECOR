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
    "batch-size": 64,
    "loss": "BCELoss",
    "learning-rate-decay-delta": 5000,
    "learning-rate-decay": 5000,
    "weight-decay": 1e-5,
    "evaluation-delta": 5,
    "num-iter": 5000,
    "num-nodes": 16,
    "momentum": 0,
    "num-labels": 2,
    "delta": 1e-5,
    "criterion": "libsvm_topk",
    "privacy": "user",
    "metric": "Loss",
    "hetero": False,
    "gradient-descent": False
    }

# Hyperparameters to test
models = ["libsvm_model"]
topologies = [("centralized", "cdp"), ("centralized", "ldp") ,("grid", "ldp"), ("ring", "ldp"), ("grid", "corr"), ("ring", "corr")]
alphas = [10.]
epsilons = [3, 5, 7, 10, 15, 20, 25, 30, 40]
# Hyperparameters for each algorithm: the true values are those of 3 and 20, the tohers I just replaced them with the closest value (3 or 20) to them
hyperparam_dict = {("centralized", "cdp", 1) : (0.05, 0.01), ("centralized", "cdp", 3): (0.05, 0.01), ("centralized", "cdp", 5): (0.05, 0.01), ("centralized", "cdp", 7):(0.05, 0.01), ("centralized", "cdp", 10):(0.05, 0.01), ("centralized", "cdp", 15): (0.01, 0.1), ("centralized", "cdp", 20): (0.01, 0.1), ("centralized", "cdp", 25):(0.1, 0.01), ("centralized", "cdp", 30): (0.01, 0.1), ("centralized", "cdp", 40): (0.01, 0.1),
                   ("centralized", "ldp", 1) : (0.005, 0.001), ("centralized", "ldp", 3): (0.005, 0.001), ("centralized", "ldp", 5): (0.005, 0.001), ("centralized", "ldp", 7): (0.005, 0.001), ("centralized", "ldp", 10): (0.005, 0.001), ("centralized", "ldp", 15): (0.05, 0.01), ("centralized", "ldp", 20): (0.05, 0.01), ("centralized", "ldp", 25): (0.05, 0.01), ("centralized", "ldp", 30): (0.05, 0.01), ("centralized", "ldp", 40): (0.05, 0.01),
                   ("grid", "ldp", 1) : (0.005, 0.001), ("grid", "ldp", 3): (0.005, 0.001), ("grid", "ldp", 5): (0.005, 0.001), ("grid", "ldp", 7): (0.005, 0.001), ("grid", "ldp", 10): (0.005, 0.001), ("grid", "ldp", 15): (0.05, 0.01), ("grid", "ldp", 20): (0.05, 0.01), ("grid", "ldp", 25): (0.05, 0.01), ("grid", "ldp", 30): (0.05, 0.01), ("grid", "ldp", 40): (0.05, 0.01),
                   ("ring", "ldp", 1) : (0.005, 0.001), ("ring", "ldp", 3): (0.005, 0.001), ("ring", "ldp", 5): (0.005, 0.001), ("ring", "ldp", 7): (0.005, 0.001), ("ring", "ldp", 10): (0.005, 0.001), ("ring", "ldp", 15): (0.05, 0.01), ("ring", "ldp", 20): (0.05, 0.01), ("ring", "ldp", 25): (0.05, 0.01), ("ring", "ldp", 30): (0.05, 0.01), ("ring", "ldp", 40): (0.05, 0.01),
                   ("grid", "corr", 1) : (0.01, 0.001), ("grid", "corr", 3): (0.01, 0.001), ("grid", "corr", 5): (0.01, 0.001), ("grid", "corr", 7): (0.01, 0.001), ("grid", "corr", 10): (0.01, 0.001), ("grid", "corr", 15): (0.001, 0.01), ("grid", "corr", 20): (0.001, 0.01), ("grid", "corr", 25): (0.001, 0.01), ("grid", "corr", 30): (0.001, 0.01), ("grid", "corr", 40): (0.001, 0.01),
                   ("ring", "corr", 1) : (0.01, 0.001), ("ring", "corr", 3): (0.01, 0.001), ("ring", "corr", 5): (0.01, 0.001), ("ring", "corr", 7): (0.01, 0.001), ("ring", "corr", 10): (0.01, 0.001), ("ring", "corr", 15): (0.001, 0.01), ("ring", "corr", 20): (0.001, 0.01), ("ring", "corr", 25): (0.001, 0.01), ("ring", "corr", 30): (0.001, 0.01), ("ring", "corr", 40): (0.001, 0.01)
}

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
                    W = topology.FixedMixingMatrix(topology_name = topology_name, n_nodes = params["num-nodes"])
                    adjacency_matrix = np.array(W(0) != 0, dtype=float)
                    adjacency_matrix = adjacency_matrix - np.diag(np.diag(adjacency_matrix))
                    degree_matrix = np.diag(adjacency_matrix @ np.ones_like(adjacency_matrix[0]))
                    eps_iter = dp_account.reverse_eps(eps= target_eps, num_iter = params["num-iter"], delta = params["delta"], subsample = 1.)

                    # sigma_cdp and sigma_ldp
                    params["learning-rate"], params["gradient-clip"] = hyperparam_dict[topology_name, method, target_eps]
                    sigma_ldp = params["gradient-clip"] * np.sqrt(2 / eps_iter)
                    sigma_cdp = sigma_ldp / np.sqrt(params["num-nodes"])

                    if "corr" in method: # CD-SGD
                        # Determining the couples (sigma, sigma_cor) that can be considered
                        df = pd.DataFrame(columns = ["topology", "sigma", "sigma-cor", "epsilon", "sigma-cdp", "sigma-ldp"])
                        sigma_grid = np.linspace(sigma_cdp, sigma_ldp, 50)
                        sigma_cor_grid = np.linspace(sigma_cdp / 1000, sigma_ldp * 10, 1000)
                        for sigma in sigma_grid:
                            all_sigma_cor = plotting.find_sigma_cor(sigma, sigma_cor_grid, params["gradient-clip"], degree_matrix, adjacency_matrix, eps_iter)
                            # check non-emptyness and add it
                            if len(all_sigma_cor) !=0:
                                eps = dp_account.rdp_compose_convert(params["num-iter"], params["delta"], sigma, all_sigma_cor[0], 
                                                                     params["gradient-clip"], degree_matrix, adjacency_matrix, subsample =1)
                                new_row = {"topology": topology_name,
                                           "sigma": sigma,
                                           "sigma-cor": all_sigma_cor[0],
                                           "epsilon": eps,
                                           "sigma-cdp": sigma_cdp,
                                           "sigma-ldp": sigma_ldp}
                                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

                        # Taking the values on the last row (correspond to the least sigma_cor): 
                        params["sigma"] = df.iloc[-1]["sigma"]
                        params["sigma-cor"] = df.iloc[-1]["sigma-cor"]
                        
                        # Store result
                        filename= f"result_gridsearch_user-level_{topology_name}_epsilon_{target_eps}.csv"
                        df.to_csv(filename)
                        jobs.submit(f"{dataset}-{topology_name}-{method}-n_{params['num-nodes']}-model_{model}-alpha_{alpha}-eps_{target_eps}", make_command(params))

                    elif "ldp" in method: # LDP
                        # To adapt with the result of the tuning
                        params["sigma-cor"] = 0
                        params["sigma"] = sigma_ldp
                        #tools.success("Submitting LDP")
                        jobs.submit(f"{dataset}-{topology_name}-{method}-n_{params['num-nodes']}-model_{model}-alpha_{alpha}-eps_{target_eps}", make_command(params))
                    
                    else: # CDP
                        # To adapt with the result of the tuning
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

# Plot Loss VS iterations
with tools.Context("libsvm", "info"):
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
                        color = method_to_color[method], lalp=0.8, logscale = True)
                        #legend.append(f"{topology_name} + {method}")
                        if topology_name not in legend_topos:
                            legend_topos.append(topology_name)
                        if method not in legend_methods:
                            legend_methods.append(method)
                    
                    # Making the legend
                    legend = []
                    legend.append(plt.Line2D([], [], label='Algorithm', linestyle = 'None'))
                    for method in legend_methods:
                        legend.append(plt.Line2D([], [], label=method.upper(), color = method_to_color[method]))
                    legend.append(plt.Line2D([], [], label='Topology', linestyle = 'None'))
                    for topo in legend_topos:
                        legend.append(plt.Line2D([], [], label= topo.capitalize(), linestyle = topo_to_style[topo], color = 'k'))
                        
                    #JS: plot every time graph in terms of the maximum number of steps
                    plot_name = f"{dataset}_model= {model}_momentum={params['momentum']}_alpha={alpha}_eps={target_eps}"
                    plot.finalize(None, "Step number", "$ \mathcal{L} - \mathcal{L}^*$", xmin=0, xmax=params['num-iter'], legend=legend)
                    plot.save(plot_directory + "/" + plot_name + ".pdf", xsize=3, ysize=1.5)

# Plot Loss VS Epsilon
# Checked !

with tools.Context("libsvm", "info"):
    for alpha in alphas:
        for model in models:
            plot = study.LinePlot()
            legend_topos = []
            legend_methods = []
            for topology_name, method in topologies:
                if topology_name not in legend_topos:
                    legend_topos.append(topology_name)
                if method not in legend_methods:
                    legend_methods.append(method)

                values = pd.DataFrame(columns = ["epsilon", params["metric"], params["metric"] +"-err"])
                for target_eps in epsilons:
                    name = f"{dataset}-{topology_name}-{method}-n_{params['num-nodes']}-model_{model}-alpha_{alpha}-eps_{target_eps}"
                    df = misc.compute_avg_err_op(name, seeds, result_directory, "eval", (params["metric"], "max"))[0]
                    new_row = {"epsilon": target_eps,
                                params["metric"]: df.iloc[-1][params["metric"]],
                                params["metric"] +"-err" : df.iloc[-1][params["metric"] +"-err"]}
                    values = pd.concat([values, pd.DataFrame([new_row])], ignore_index=True)
                    

                plot.include(values, params["metric"], errs="-err", xticks = epsilons, linestyle = topo_to_style[topology_name], 
                                    mark = method_to_marker[method], color = method_to_color[method], lalp=0.8, logscale = True)
            # Making the legend
            legend = []
            legend.append(plt.Line2D([], [], label='Algorithm', linestyle = 'None' ))
            for method in legend_methods:
                legend.append(plt.Line2D([], [], label=method.upper(), color = method_to_color[method], marker = method_to_marker[method]))
            legend.append(plt.Line2D([], [], label='Topology', linestyle = 'None'))
            for topo in legend_topos:
                legend.append(plt.Line2D([], [], label= topo.capitalize(), linestyle = topo_to_style[topo], color = 'k'))

            #JS: plot every time graph in terms of the maximum number of steps
            plot_name = f"Loss_vs_epsilon_{dataset}_model={model}_momentum={params['momentum']}_alpha={alpha}"
            plot.finalize(None, "Step number", "Test Loss", legend = legend)
            plot.save(plot_directory + "/" + plot_name + ".pdf", xsize=3, ysize=1.5)
