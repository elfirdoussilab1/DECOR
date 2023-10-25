# coding: utf-8
###
 # @file   reproduce_mnist.py
 # @author Aymane El Firdoussi <aymane.elfirdoussi@epfl.ch>
 #
 # Running Correlated Decentralized learning experiments on MNIST.
###
import utils
from utils import *
from utils import tools, study, jobs, dp_account
utils.success("Module loading...")
import signal, torch
import pandas as pd

# ---------------------------------------------------------------------------- #
# Miscellaneous initializations
utils.success("Miscellaneous initializations...")

# "Exit requested" global variable accessors
exit_is_requested, exit_set_requested = utils.onetime("exit")

# Signal handlers
signal.signal(signal.SIGINT, exit_set_requested)
signal.signal(signal.SIGTERM, exit_set_requested)

# ---------------------------------------------------------------------------- #

#JS: Pick the dataset on which to run experiments
dataset = "mnist"
result_directory = "results-data-" + dataset 
plot_directory = "results-plot-" + dataset

with utils.Context("cmdline", "info"):
    args = tools.process_commandline()
    # Make the result directories
    args.result_directory = check_make_dir(result_directory)
    args.plot_directory = check_make_dir(plot_directory)
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
utils.success("Running experiments...")

# Base parameters for the MNIST experiments
params = {
    "dataset": "mnist",
    "batch-size": 25,
    "loss": "NLLLoss",
    "learning-rate-decay-delta": 50,
    "learning-rate-decay": 50,
    "weight-decay": 1e-4,
    "evaluation-delta": 5,
    "gradient-clip": 2,
    "num-iter": 1000,
    "num-nodes": 16,
    "momentum": 0.9,
    "num-labels": 10,
    "delta": 1e-4
    }

# Hyperparameters to test
models = [("cnn_mnist", 0.75)]
topologies = [("centralized", "cdp") ,("ring", "correlated"), ("ring", "ldp")]
alphas = [0.1]
sigmas_cdp = [5]
sigmas_cor = [14]


# Command maker helper
def make_command(params):
    cmd = ["python3", "-OO", "train.py"]
    cmd += utils.dict_to_cmdlist(params)
    return utils.Command(cmd)

# Jobs
jobs  = jobs.Jobs(args.result_directory, devices=args.devices, devmult=args.supercharge)
seeds = jobs.get_seeds()

# Submit all experiments
for alpha in alphas:
    for model, lr in models:
        for sigma_cdp in sigmas_cdp:
             for sigma_cor in sigmas_cor:
                for topology, method in topologies:
                    params["model"] = model
                    params["learning-rate"] = lr
                    params["dirichlet-alpha"] = alpha
                    params["topology"] = topology
                    if "ldp" not in method: #CDP or correlated
                        params["sigma-cdp"] = sigma_cdp
                        params["sigma-cor"]= sigmas_cor
                        jobs.submit(f"{dataset}-{topology}-{method}-n_{params['num-nodes']}-model_{model}-lr_{lr}-momentum_{params['momentum']}-sigma-cdp_{sigma_cdp}-sigma-cor_{sigma_cor}-alpha_{alpha}", make_command(params))
                    else: # LDP
                        params["sigma-cor"] = 0
                        W = topology.FixedMixingMatrix(topology, params["num-nodes"])
                        adjacency_matrix = np.array(W(0) != 0, dtype=float)
                        adjacency_matrix = adjacency_matrix - np.diag(np.diag(adjacency_matrix))
                        degree_matrix = np.diag(adjacency_matrix @ np.ones_like(adjacency_matrix[0]))

                        eps = dp_account.rdp_account(sigma_cdp, sigma_cor, params["gradient-clip"], degree_matrix, adjacency_matrix)
                        params["eps-iter"] = eps
                        params["sigma-cdp"] = params["gradient-clip"] * np.sqrt(2/eps)
                        print(f"User-Privacy {dp_account.rdp_compose_convert(params['num-iter'], params['eps-iter'], params['delta'])}")
                        jobs.submit(f"{dataset}-{topology}-{method}-n_{params['num-nodes']}-model_{model}-lr_{lr}-momentum_{params['momentum']}-sigma-cdp_{sigma_cdp}-sigma-cor_{sigma_cor}-alpha_{alpha}", make_command(params))
                           

# Wait for the jobs to finish and close the pool
jobs.wait(exit_is_requested)
jobs.close()

# Check if exit requested before going to plotting the results
if exit_is_requested():
    exit(0)

 # ---------------------------------------------------------------------------- #
 # Plot results
utils.success("Plotting results...")


# Plot results without subsampling
with utils.Context("mnist", "info"):
    for alpha in alphas:
        for model, lr in models:
            for sigma_cdp in sigmas_cdp:
                for sigma_cor in sigmas_cor:
                    values = dict()
                    # Plot top-1 cross-accuracies
                    plot = study.LinePlot()
                    legend = []
                    for topology, method in topologies:
                        name = f"{dataset}-{topology}-{method}-n_{params['num-nodes']}-model_{model}-lr_{lr}-
                        momentum_{params['momentum']}-sigma-cdp_{sigma_cdp}-sigma-cor_{sigma_cor}-alpha_{alpha}"
                        values[topology, method] = tools.compute_avg_err_op(name, seeds, result_directory, "eval", ("Accuracy", "max"))
                        plot.include(values[topology, method][0], "Accuracy", errs="-err", lalp=0.8)
                        legend.append(f"{topology} + {method}")

                    #JS: plot every time graph in terms of the maximum number of steps
                    plot_name = f"{dataset} _model= {model} _lr= {lr}_sigma_cdp= {sigma_cdp}_sigma_cor={sigma_cor}_alpha={alpha}"
                    plot.finalize(None, "Step number", "Test accuracy", xmin=0, xmax=params['num-iter'], ymin=0, ymax=1, legend=legend)
                    plot.save(plot_directory + "/" + plot_name + ".pdf", xsize=6, ysize=2)
