import torch, random, argparse, sys, os
import numpy as np
import pandas as pd
from utils import study


# Flatten list of tensors. Used for model parameters and gradients
def flatten(list_of_tensors):
    return torch.cat(tuple(tensor.view(-1) for tensor in list_of_tensors))

# Unflatten a flat tensor. Used when setting model parameters and gradients
def unflatten(flat_tensor, model_shapes):
    c = 0
    returned_list = [torch.zeros(shape) for shape in model_shapes]
    for i, shape in enumerate(model_shapes):
        count = 1
        for element in shape:
            count *= element
        returned_list[i].data = flat_tensor[c:c + count].view(shape)
        c = c + count
    return returned_list

# Apply Gradient clipping
def clip_vector(vector, c_clip):
    vector_norm = vector.norm().item()
    if vector_norm > c_clip:
        vector.mul_(c_clip / vector_norm)
    return vector

# Fix seed
def fix_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# ---------------------------------------------------------------------------- #
# Criterions to evaluate accuracy of models. Used in worker.py and p2pWorker.py

def topk(output, target, k=1):
      """ Compute the top-k criterion from the output and the target.
      Args:
        output Batch × model logits
        target Batch × target index
      Returns:
        1D-tensor [#correct classification, batch size]
      """
      res = (output.topk(k, dim=1)[1] == target.view(-1).unsqueeze(1)).any(dim=1).sum()
      return torch.cat((res.unsqueeze(0), torch.tensor(target.shape[0], dtype=res.dtype, device=res.device).unsqueeze(0)))

# ---------------------------------------------------------------------------- #
# Functions used for dataset manipulation in dataset.py

# Returns the indices of the training datapoints selected for each honest worker, in case of Dirichlet distribution
def draw_indices(samples_distribution, indices_per_label, nb_workers):
    
    # Initialize the dictionary of samples per worker. Should hold the indices of the samples each worker possesses
    worker_samples = dict()
    for worker in range(nb_workers):
        worker_samples[worker] = list()

    for label, label_distribution in enumerate(samples_distribution):
        last_sample = 0
        number_samples_label = len(indices_per_label[label])
        # Iteratively split the number of samples of label into chunks according to the worker proportions, and assign each chunk to the corresponding worker
        for worker, worker_proportion in enumerate(label_distribution):
            samples_for_worker = int(worker_proportion * number_samples_label)
            worker_samples[worker].extend(indices_per_label[label][last_sample:last_sample+samples_for_worker])
            last_sample = samples_for_worker

    return worker_samples

# Function used to parse the command-line and perform checks in the reproduce scripts
def process_commandline():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--result-directory",
        type=str,
        default="results-data",
        help="Path of the data directory, containing the data gathered from the experiments")
    parser.add_argument("--plot-directory",
        type=str,
        default="results-plot",
        help="Path of the plot directory, containing the graphs traced from the experiments")
    parser.add_argument("--devices",
        type=str,
        default="auto",
        help="Comma-separated list of devices on which to run the experiments, used in a round-robin fashion")
    parser.add_argument("--supercharge",
        type=int,
        default=1,
        help="How many experiments are run in parallel per device, must be positive")
    # Parse command line
    return parser.parse_args(sys.argv[1:])

#JS: Function used to plot the results of the experiments in the reproduce scripts
def compute_avg_err_op(name, seeds, result_directory, location, *colops, avgs="", errs="-err"):
    """ Compute the average and standard deviation of the selected columns over the given experiment.
    Args:
        name Given experiment name
        seeds   Seeds used for the experiment
        result_directory Directory to store the results
        location Script to read from
        ...  Tuples of (selected column name (through 'study.select'), optional reduction operation name)
        avgs Suffix for average column names
        errs Suffix for standard deviation (or "error") column names
    Returns:
        Data frames for each of the computed columns,
        Tuple of reduced values per seed (or None if None was provided for 'op')
    Raises:
        'RuntimeError' if a reduction operation was specified for a column selector that did not select exactly 1 column
    """
    # Load all the runs for the given experiment name, and keep only a subset
    datas = tuple(study.select(study.Session(result_directory + "/" + name + "-" +str(seed), location), *(col for col, _ in colops)) for seed in seeds)

    # Make the aggregated data frames
    def make_df_ro(col, op):
        nonlocal datas
        # For every selected columns
        subds = tuple(study.select(data, col).dropna() for data in datas)
        df    = pd.DataFrame(index=subds[0].index)
        ro    = None
        for cn in subds[0]:
            # Generate compound column names
            avgn = cn + avgs
            errn = cn + errs
            # Compute compound columns
            numds = np.stack(tuple(subd[cn].to_numpy() for subd in subds))
            df[avgn] = numds.mean(axis=0)
            df[errn] = numds.std(axis=0)
            # Compute reduction, if requested
            if op is not None:
                if ro is not None:
                    raise RuntimeError(f"column selector {col!r} selected more than one column ({(', ').join(subds[0].columns)}) while a reduction operation was requested")
                ro = tuple(getattr(subd[cn], op)().item() for subd in subds)
        # Return the built data frame and optional computed reduction
        return df, ro
    dfs = list()
    ros = list()
    for col, op in colops:
        df, ro = make_df_ro(col, op)
        dfs.append(df)
        ros.append(ro)
    # Return the built data frames and optional computed reductions
    return dfs

# ---------------------------------------------------------------------------- #
# Fuction used in train.py

# Print the configuration of the current training in question
def print_conf(subtree, level=0):
    if isinstance(subtree, tuple) and len(subtree) > 0 and isinstance(subtree[0], tuple) and len(subtree[0]) == 2:
        label_len = max(len(label) for label, _ in subtree)
        iterator  = subtree
    elif isinstance(subtree, dict):
        if len(subtree) == 0:
            return " - <none>"
        label_len = max(len(label) for label in subtree.keys())
        iterator  = subtree.items()
    else:
        return f" - {subtree}"
    level_spc = "  " * level
    res = ""
    for label, node in iterator:
        res += f"{os.linesep}{level_spc}· {label}{' ' * (label_len - len(label))}{print_conf(node, level + 1)}"
    return res