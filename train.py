# Main file for experiments
import torch, argparse, sys, signal, pathlib, os
from worker import *
import dataset, utils, models
import worker
utils.success("Module loading...")
from utils import tools, topology
import numpy as np

# "Exit requested" global variable accessors
exit_is_requested, exit_set_requested = tools.onetime("exit")


# ---------------------------------------------------------------------------- #
# Setup
utils.success("Experiment setup...")

# Signal handlers
signal.signal(signal.SIGINT, exit_set_requested)
signal.signal(signal.SIGTERM, exit_set_requested)

# ---------------------------------------------------------------------------- #
# Command-line processing
tools.success("Command-line processing...")

def process_commandeline():
    """ Parse the command-line and perform checks
    Returns:
        Parsed configuration
    """
    # Description
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--seed",
        type= int,
        default= -1,
        help="Fixed seed for reproducibility, negative for random seed")
    parser.add_argument("--device",
        type = str,
        default= "auto",
        help = "Device on which to run the experiment, auto by default")
    parser.add_argument("--num-iter",
        type = int,
        default= 1000,
        help = "Total number of iterations for learning")
    parser.add_argument("--num-nodes",
        type = int,
        default= 16,
        help = "Number of nodes")
    parser.add_argument("--topology",
        type = str,
        default=None,
        help = "Topology of the network")
    parser.add_argument("--method",
        type = str,
        default=None,
        help = "Argument used to differentiate between LDP and Correlated SGD")
    parser.add_argument("--sigma-cdp",
        type = float,
        default= 0,
        help = "CDP Noise")
    parser.add_argument("--sigma-cor",
        type = float,
        default= 0,
        help = "Correlated Noise")
    parser.add_argument("--delta",
        type = float,
        default= 1e-4,
        help = "Second parameter of Privacy")
    parser.add_argument("--model",
        type = str,
        default=None,
        help = "Model to use in training")
    parser.add_argument("--loss",
        type = str,
        default="NLLLoss",
        help = "Loss to use")
    parser.add_argument("--criterion",
        type = str,
        default="topk",
        help = "Evaluation Criterion")
    parser.add_argument("--dataset",
        type = str,
        default= None,
        help = "Dataset to use")
    parser.add_argument("--batch-size",
        type = int,
        default= 25,
        help = "Training batch size")
    parser.add_argument("--batch-size-test",
        type = int,
        default= 100,
        help ="Test batch size")
    parser.add_argument("--learning-rate",
        type = float,
        default= None,
        help = "Learning rate to use for tratining")
    parser.add_argument("--learning-rate-decay",
        type = int,
        default= 5000,
        help = "Learning rate hyperbolic half-decay time, non-positive for no decay")
    parser.add_argument("--learning-rate-decay-delta",
        type = int,
        default=1,
        help = "How many steps between two learning rate updates")
    parser.add_argument("--momentum",
        type = float,
        default= 0.9,
        help ="Momentum")
    parser.add_argument("--weight-decay",
        type = float,
        default= 0,
        help = "Weight decay (L2 regularization)")
    parser.add_argument("--gradient-clip",
        type = float,
        default= None,
        help = "Gradient clippin threshold")
    parser.add_argument("--result-directory",
        type = str, 
        default=None,
        help = "Path of the directory in which to save the experiment results and checkpoints")
    parser.add_argument("--evaluation-delta",
        type = int,
        default= 5,
        help = "Interval of iterations between each evaluation")
    parser.add_argument("--num-evaluations",
        type = int,
        default= 100, 
        help = "Number of evaluations in testing phase")
    parser.add_argument("--num-labels",
        type = int,
        default=None,
        help = "Number of labels in the dataset")
    parser.add_argument("--hetero",
        action = "store_true",
        default= False,
        help = "Heterogeneous setting")
    parser.add_argument("--dirichlet-alpha",
        type = float,
        default= False,
        help = "The parameter of Dirichlet distribution")
    parser.add_argument("--gradient-descent",
        action = "store_true",
        default= None,
        help = "Execute the full gradient descent algorithm")

    return parser.parse_args(sys.argv[1:])

with utils.Context("cmdline", "info"):
    args = process_commandeline()
    cmdline_config = "Configuration" + tools.print_conf((

        ("Reproducibility", "not enforeced" if args.seed < 0 else (f"enforced (seed {args.seed})")),
        ("Number of nodes", args.num_nodes),
        ("Model", args.model),
        ("Dataset", (
            ("Name", args.dataset),
            ("Batch size", (
                ("Training", args.batch_size),
                ("Testing", args.batch_size_test))))),
        ("Topology", args.topology),
        ("CDP Noise", args.sigma_cdp),
        ("Correlated Noise", args.sigma_cor),
        ("Loss", (
            ("Name", args.loss),
            ("L2-regularization", "none" if args.weight_decay is None else f"{args.weight_decay}"))),
        ("Criterion", args.criterion),
        ("Optimizer", (
            ("Name", "sgd"),
            ("Learning rate", args.learning_rate),
            ("Momentum", f"{args.momentum_worker}"))),
        ("Gradient clip", "no" if args.gradient_clip is None else f"{args.gradient_clip}"),
        ("Extreme Heterogeneity", "yes" if args.hetero else "no"),
        ("Dirichlet distribution", "alpha = " + str(args.dirichlet_alpha) if args.dirichlet_alpha is not None else "no"),
        ("Gradient descent", "yes" if args.gradient_descent else "no")))
    
    print(cmdline_config)
    

def result_make(name, fields):
    """ Make and bind a new result file with a name, initialize with a header line.
    Args:
        name    Name of the result file
        fields  List of names of each field
    Raises:
        'KeyError' if name is already bound
        'RuntimeError' if no name can be bound
        Any exception that 'io.FileIO' can raise while opening/writing/flushing
    """
    # Check if results are to be output
    global args
    if args.result_directory is None:
        raise RuntimeError("No result is to be output")
    # Check if name is already bounds
    global result_fds
    if name in result_fds:
        raise KeyError(f"Name {name!r} is already bound to a result file")
    # Make the new file
    fd = (args.result_directory / name).open("w")
    fd.write("# " + ("\t").join(str(field) for field in fields))
    fd.flush()
    result_fds[name] = fd

def result_get(name):
    """ Get a valid descriptor to the bound result file, or 'None' if the given name is not bound.
    Args:
        name Given name
    Returns:
        Valid file descriptor, or 'None'
    """
    # Check if results are to be output
    global args
    if args.result_directory is None:
        return None
    # Return the bound descriptor, if any
    global result_fds
    return result_fds.get(name, None)

def result_store(fd, entries):
    """ Store a line in a valid result file.
    Args:
        fd     Descriptor of the valid result file
        entries... List of Object(s) to convert to string and write in order in a new line
    """
    fd.write(os.linesep + ("\t").join(str(entry) for entry in entries))
    fd.flush()

with utils.Context("setup", "info"):
    tools.fix_seed(1)
    # Create train and test loaders
    train_loader_dict, test_loader = dataset.make_train_test_datasets(dataset=args.dataset, gradient_descent= args.gradient_descent, heterogeneity=args.hetero,
                                  num_labels=args.num_labels, alpha_dirichlet= args.dirichlet_alpha, num_nodes=args.num_nodes, train_batch=args.batch_size, test_batch=args.batch_size_test)
    
    reproducible = (args.seed >= 0)
    if reproducible:
        tools.fix_seed(args.seed)
    torch.backends.cudnn.deterministic = reproducible
    torch.backends.cudnn.benchmark   = not reproducible

    # Make the result directory if requested 
    if args.result_directory is not None:
        try:
            resdir = pathlib.Path(args.result_directory).resolve()
            resdir.mkdir(mode=0o755, parents=True, exist_ok=True)
            args.result_directory = resdir
        except Exception as err:
            utils.warning(f"Unable to create the result directory {str(resdir)!r} ({err}); no result will be stored")
        result_fds = dict()
        # Make evaluation file
        if args.evaluation_delta > 0:
            result_make("eval", "Step number", "Mean-accuracy")

# ---------------------------------------------------------------------------- #
# Training
utils.success("Training...")

def update_learning_rate(step, lr, args):
    if args.learning_rate_decay > 0 and step % args.learning_rate_decay_delta == 0:
        return lr / (step / args.learning_rate_decay + 1)
    else:
        return lr

with utils.Context("training", "info"):
    was_training = False
    fd_eval = result_get("eval")

    # Agree on the initial parameters
    model = getattr(models, args.model)()
    model.to(args.device)
    initial_parameters = tools.flatten(model.parameters())
    model_size = len(initial_parameters)

    # Initialize Workers
    workers = []
    for i in range(args.num_nodes):
        data_loader = train_loader_dict[i]
        worker_i = worker.Worker(train_data_loader=data_loader, test_data_loader=test_loader, batch_size=args.batch_size, 
                    model = args.model, loss = args.loss, momentum = args.momentum, gradient_clip= args.gradient_clip, sigma_cdp= args.sigma_cdp,
                    num_labels= args.num_labels, criterion= args.criterion, num_evaluations= args.num_evaluations, device= args.device)
        workers.append(worker_i)
    
    # Weights matrix
    W = topology.FixedMixingMatrix(topology_name= args.topology, n_nodes= args.num_nodes)

    # Convert it to tensor
    W = torch.tensor(W)
    
    # Noise tensor: shape (num_nodes, num_nodes, model_size)
    V = torch.randn(args.num_nodes, args.num_nodes, model_size) # distribution N(0, 1)
    V.mul_(args.sigma_cor) # rescaling ==> distribution N (0, sigma_cor^2)

    # Antisymmetry property
    tools.to_antisymmetric(V)
    
    # Initializing learning rate
    lr = args.learning_rate

    # ------------------------------------------------------------------------ #
    current_step = 0
    while not exit_is_requested() and current_step <= args.num_iter:
        # Update the learning rate
        lr = update_learning_rate(current_step, lr, args)
        
        # Evaluate the model if milestone is reached
        milestone_evaluation = args.evaluation_delta > 0 and current_step % args.evaluation_delta == 0        
        if milestone_evaluation:
            mean_accuracy = np.mean([workers[i].compute_accuracy() for i in range(args.num_nodes)])
            print(f"Mean Accuracy (step {current_step})... {mean_accuracy * 100.:.2f}%.")
            # Store the evaluation result
            if fd_eval is not None:
                result_store(fd_eval, [current_step, mean_accuracy])
        
        # Apply the algorithm
        all_parameters = []

        # Step t + 1/2
        for i in range(args.num_nodes):
            workers[i].compute_momentum()
            all_parameters.append(workers[i].grad_descent(V[i], lr = lr, weight_decay = args.weight_decay))
        
        all_parameters = torch.tensor(all_parameters)
        
        # Step t + 1
        for i in range(args.num_nodes):
            workers[i].decentralized_learning(weights = W[i], workers_parameters = all_parameters)

        current_step += 1

utils.sucess("Finished...")


