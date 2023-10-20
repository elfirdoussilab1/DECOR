import torch
import numpy as np

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