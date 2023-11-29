# Main file for experiments
import torch, argparse, sys, signal, pathlib, os
from worker import *
import dataset, tools, models, misc, evaluator
import worker
tools.success("Module loading...")
from utils import topology
import numpy as np

# "Exit requested" global variable accessors
exit_is_requested, exit_set_requested = tools.onetime("exit")


# ---------------------------------------------------------------------------- #
# Setup
tools.success("Experiment setup...")

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
    parser.add_argument("--topology-name",
        type = str,
        default=None,
        help = "Topology of the network")
    parser.add_argument("--method",
        type = str,
        default=None,
        help = "Argument used to differentiate between LDP and Correlated SGD")
    parser.add_argument("--sigma",
        type = float,
        default= 0,
        help = "CDP Noise")
    parser.add_argument("--sigma-cor",
        type = float,
        default= 0,
        help = "Correlated Noise")
    parser.add_argument("--epsilon",
        type = float,
        default= 0,
        help = "User-rpvacy")
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
        default=None,
        help = "Loss to use")
    parser.add_argument("--criterion",
        type = str,
        default="topk",
        help = "Evaluation Criterion")
    parser.add_argument("--metric",
        type = str,
        default="accuracy",
        help = "Evaluation to plot, either: Loss (LibSVM) or Accuracy (torchvision datasets)")
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
        default= 0.,
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
    parser.add_argument("--privacy",
        type = str,
        default= None,
        help = "type of privacy: either user or example")

    return parser.parse_args(sys.argv[1:])

with tools.Context("cmdline", "info"):
    args = process_commandeline()
    cmdline_config = "Configuration" + misc.print_conf((

        ("Reproducibility", "not enforeced" if args.seed < 0 else (f"enforced (seed {args.seed})")),
        ("Number of nodes", args.num_nodes),
        ("Model", args.model),
        ("Dataset", (
            ("Name", args.dataset),
            ("Batch size", (
                ("Training", args.batch_size),
                ("Testing", args.batch_size_test))))),
        ("Topology", args.topology_name),
        ("Noise", args.sigma),
        ("Correlated Noise", args.sigma_cor),
        ("Loss", (
            ("Name", args.loss),
            ("L2-regularization", "none" if args.weight_decay is None else f"{args.weight_decay}"))),
        ("Criterion", args.criterion),
        ("Optimizer", (
            ("Name", "sgd"),
            ("Learning rate", args.learning_rate),
            ("Momentum", f"{args.momentum}"))),
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

with tools.Context("setup", "info"):
    misc.fix_seed(1)
    # Create train and test loaders
    train_loader_dict, test_loader = dataset.make_train_test_datasets(dataset=args.dataset, gradient_descent= args.gradient_descent, heterogeneity=args.hetero,
                                  num_labels=args.num_labels, alpha_dirichlet= args.dirichlet_alpha, num_nodes=args.num_nodes, train_batch=args.batch_size, test_batch=args.batch_size_test)
    
    # Create the evaluator
    server = evaluator.Evaluator(train_loader_dict, test_loader, model = args.model, loss = args.loss, num_labels= args.num_labels, criterion = args.criterion, num_evaluations= args.num_evaluations, 
                                 device=args.device)

    reproducible = (args.seed >= 0)
    if reproducible:
        misc.fix_seed(args.seed)
    torch.backends.cudnn.deterministic = reproducible
    torch.backends.cudnn.benchmark   = not reproducible

    # Make the result directory if requested 
    if args.result_directory is not None:
        try:
            resdir = pathlib.Path(args.result_directory).resolve()
            resdir.mkdir(mode=0o755, parents=True, exist_ok=True)
            args.result_directory = resdir
        except Exception as err:
            tools.warning(f"Unable to create the result directory {str(resdir)!r} ({err}); no result will be stored")
        result_fds = dict()
        # Make evaluation file
        if args.evaluation_delta > 0:
            if "loss" in args.metric:
                result_make("eval", ["Step number", "Loss"])
            else:
                result_make("eval", ["Step number", "Accuracy"])
        result_make("track", ["Step number", "topology", "method", "lr", "clip", "sigma", "sigma_cor"])

# ---------------------------------------------------------------------------- #
# Training
tools.success("Training...")

def update_learning_rate(step, lr, args):
    if args.learning_rate_decay > 0 and step % args.learning_rate_decay_delta == 0:
        return args.learning_rate / (step / args.learning_rate_decay + 1)
    else:
        return lr

with tools.Context("training", "info"):

    fd_eval = result_get("eval")
    fd_track = result_get("track")


    # Initialize Workers
    workers = []
    for i in range(args.num_nodes):
        data_loader = train_loader_dict[i]
        worker_i = worker.Worker(train_data_loader=data_loader, test_data_loader=test_loader, batch_size=args.batch_size, 
                    model = args.model, loss = args.loss, momentum = args.momentum, gradient_clip= args.gradient_clip, sigma= args.sigma,
                    num_labels= args.num_labels, criterion= args.criterion, num_evaluations= args.num_evaluations, device= args.device,
                    privacy= args.privacy)
        # Agree on first parameters
        worker_i.flat_parameters = server.flat_parameters
        worker_i.update_model_parameters()
        workers.append(worker_i)
    
    # Weights matrix
    W = topology.FixedMixingMatrix(topology_name= args.topology_name, n_nodes= args.num_nodes)(0)

    # Convert it to tensor
    W = torch.tensor(W, dtype= torch.float).to(args.device)
    print(W[0])
    
    # Noise tensor: shape (num_nodes, num_nodes, model_size)
    V = torch.randn(args.num_nodes, args.num_nodes, workers[0].model_size) # distribution N(0, 1)
    V.mul_(args.sigma_cor) # rescaling ==> distribution N (0, sigma_cor^2)

    # Antisymmetry property
    V = misc.to_antisymmetric(V).to(args.device)
    print(V)
    # Initializing learning rate
    lr = args.learning_rate

    # ------------------------------------------------------------------------ #
    current_step = 0
    while not exit_is_requested() and current_step <= args.num_iter:
        
        # Evaluate the model if milestone is reached
        milestone_evaluation = args.evaluation_delta > 0 and current_step % args.evaluation_delta == 0        
        if milestone_evaluation:
            #mean_accuracy = np.mean([workers[i].compute_accuracy() for i in range(args.num_nodes)])
            mean_param = torch.stack([workers[i].flat_parameters]).mean(dim = 0)
            server.update_model_parameters(mean_param)
            mean_metric = 0
            if "Loss" in args.metric:
                mean_metric = server.compute_train_loss() - 0.3236
            else: # accuracy
                mean_metric = server.compute_accuracy()
            #mean_accuracy = workers[0].compute_accuracy()
            print(f"Mean {args.metric} (step {current_step})... {mean_metric * 100.:.2f}%.")

            # Store the evaluation result
            if fd_eval is not None:
                result_store(fd_eval, [current_step, mean_metric])
            
            if fd_track is not None:
                result_store(fd_track, [current_step, lr, args.topology_name, args.method, args.sigma, args.sigma_cor])
        
        # Update the learning rate
        lr = update_learning_rate(current_step, lr, args)

        # Apply the algorithm
        all_parameters = []

        # Step t + 1/2
        for i in range(args.num_nodes):
            all_parameters.append(workers[i].grad_descent(V[i], lr = lr, weight_decay = args.weight_decay))
        
        all_parameters = torch.stack(all_parameters).to(args.device)
        
        # Step t + 1
        for i in range(args.num_nodes):
            workers[i].decentralized_learning(weights = W[i], workers_parameters = all_parameters)

        current_step += 1

tools.success("Finished...")


