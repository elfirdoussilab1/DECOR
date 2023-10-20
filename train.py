# Main file for experiments
import torch
from worker import *
import dataset, utils, os
from utils import tools

# Parameters and hyperparameters
dataset_name = "mnist" # "cifar10"
loss = "NLLLoss"
batch_size = 25
lr = 0.75
learning_rate_decay = 50
learning_rate_decay_delta = 50
weight_decay = 1e-4
evaluation_delta = 5
c_clip = 2
num_nodes = 64
momentum = 0.9
num_labels = 10
alpha = 0.1 # Dirichlet




# devices
devices = list(f"cuda:{i}" for i in range(torch.cuda.device_count()))


# Storing results
result_directory = "results-data-" + dataset + "-mixing"
plot_directory = "results-plot-" + dataset + "-mixing"

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
    if result_directory is None:
        raise RuntimeError("No result is to be output")
    # Check if name is already bounds
    global result_fds
    if name in result_fds:
        raise KeyError(f"Name {name!r} is already bound to a result file")
    # Make the new file
    fd = (result_directory / name).open("w")
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
    if result_directory is None:
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

# 

