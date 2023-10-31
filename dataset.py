# Dataset rappers/helpers

import random, pathlib
import torch, torchvision
import torchvision.transforms as T
import numpy as np
import misc

# ---------------------------------------------------------------------------- #
# Collection of default transforms
transforms_horizontalflip = [T.RandomHorizontalFlip(), T.ToTensor()]
# Transforms from "A Little is Enough" (https://github.com/moranant/attacking_distributed_learning)
transforms_mnist = [T.ToTensor(), T.Normalize((0.1307,), (0.3081,))] 
# Transforms from https://github.com/kuangliu/pytorch-cifar
transforms_cifar = [T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))] 

# Transformations per-dataset for train and test data
transforms = {
    "mnist": (transforms_mnist, transforms_mnist),
    "fashionmnist": (transforms_horizontalflip, transforms_horizontalflip),
    "cifar10": (transforms_cifar, transforms_cifar),
    "cifar100": (transforms_cifar, transforms_cifar),
    "imagenet": (transforms_horizontalflip, transforms_horizontalflip)
}

# Dataset names in Pytorch
dataset_names = {
    "mnist": "MNIST",
    "fashionmnist": "FashionMNIST",
    "emnist":       "EMNIST",
    "cifar10":      "CIFAR10",
    "cifar100":     "CIFAR100",
    "imagenet":     "ImageNet"
}

# ---------------------------------------------------------------------------- #
# Dataset wrapper class
class Dataset:

    def __init__(self, dataset_name, num_nodes = None, train=False, gradient_descent= False, num_labels= None, heterogeneity = False,
                 alpha_dirichlet=None, batch_size=None):
        
        """ Dataset builder constructor.
        Args:
            dataset_name (str)         Dataset string name (see dataset_names dictionary)
            num_nodes (int)            Total number of nodes
            train (boolean)            Boolean that is true during training, and false during testing
            gradient_descent (boolean) Boolean that is true in the case of the full gradient descent algorithm
            num_labels (int)           Number of labels of the dataset in question
            heterogeneity (boolean)    Boolean that is true in heterogeneous setting
            alpha_dirichlet (float)    Value of parameter alpha for dirichlet distribution
            batch_size (int)           Batch size used during the training or testing
        """

        if train:
            # Load the initial training dataset
            dataset = getattr(torchvision.datasets, dataset_names[dataset_name])(root = get_default_root(), train = True, download = True,
                                                                                transform = T.Compose(transforms[dataset_name][0]))
            
            targets = dataset.targets
            if isinstance(targets, list):
                targets = torch.FloatTensor(targets)

            #JS: extreme heterogeneity setting while training
            if heterogeneity:
                labels = range(num_labels)
                ordered_indices = []
                for label in labels:
                    label_indices = (targets == label).nonzero().tolist()
                    label_indices = [item for sublist in label_indices for item in sublist]
                    ordered_indices += label_indices

                self.dataset_dict = {}

                split_indices = np.array_split(ordered_indices, num_nodes)
                for worker_id in range(num_nodes):
                    dataset_modified = torch.utils.data.Subset(dataset, split_indices[worker_id].tolist())
                    if gradient_descent:
                        #JS: Adjust batch size in case of gradient descent
                        batch_size = len(split_indices[worker_id])
                    dataset_worker = torch.utils.data.DataLoader(dataset_modified, batch_size=batch_size, shuffle=True)
                    #JS: have one dataset iterator per honest worker
                    self.dataset_dict[worker_id] = dataset_worker

            elif alpha_dirichlet is not None:
                # store in indices_per_label the list of indices of each label (0 then 1 then 2 ...)
                indices_per_label = dict()
                for label in range(num_labels):
                    label_indices = (targets == label).nonzero().tolist()
                    label_indices = [item for sublist in label_indices for item in sublist]
                    indices_per_label[label] = label_indices
                
                # compute number of samples of each worker for each class, using a Dirichlet distribution of parameter alpha_dirichlet
                samples_distribution = np.random.dirichlet(np.repeat(alpha_dirichlet, num_nodes), size=num_labels)
                print(f"Proportions for label 1 {samples_distribution[0]}")
                # get the indices of the samples belonging to each worker (stored in dict worker_samples)
                worker_samples = misc.draw_indices(samples_distribution, indices_per_label, num_nodes)
                
                self.dataset_dict = {}
                for worker_id in range(num_nodes):
                    dataset_modified = torch.utils.data.Subset(dataset, worker_samples[worker_id])
                    if gradient_descent:
                        # Adjust batch size in case of gradient descent
                        batch_size = len(worker_samples[worker_id])
                    # have one dataset iterator per honest worker
                    self.dataset_dict[worker_id] = torch.utils.data.DataLoader(dataset_modified, batch_size=batch_size, shuffle=True)

        else:
            # testing set
            dataset_test = getattr(torchvision.datasets, dataset_names[dataset_name])(root=get_default_root(), train=False, download=False,
                                                                                    transform=T.Compose(transforms[dataset_name][0]))
            self.data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)


def make_train_test_datasets(dataset, gradient_descent=False, heterogeneity=False, num_labels=None, alpha_dirichlet=None,
    num_nodes=None, train_batch=None, test_batch=None):
  """ Helper to make new instance of train and test datasets.
  Args:
    dataset (str)                Case-sensitive dataset name
    gradient_descent (boolean)   Boolean that is true in the case of the full gradient descent algorithm
    heterogeneity (boolean)      Boolean that is true in heterogeneous setting
    numb_labels (int)            Number of labels of dataset
    alpha_dirichlet (float)      Value of parameter alpha for dirichlet distribution
    num_nodes (int)              Number of honest workers in the system
    train_batch (int)            Training batch size
    test_batch (int)             Testing batch size
  Returns:
    Dictionary of training datasets for honest workers and data loader for test dataset
  """
  # Make the training dataset
  trainset = Dataset(dataset_name = dataset, num_nodes= num_nodes, train=True, gradient_descent=gradient_descent, num_labels=num_labels,
                     heterogeneity=heterogeneity, alpha_dirichlet=alpha_dirichlet, batch_size=train_batch)

  # Make the testing dataset
  testset = Dataset(dataset, train=False, batch_size=test_batch)

  # Return the data loaders
  return trainset.dataset_dict, testset.data_loader_test


def get_default_root():
    """ Lazy-initialize and return the default dataset root directory path."""
    # Generate the default path
    default_root = pathlib.Path(__file__).parent / "datasets" / "cache"
    # Create the path if it does not exist
    default_root.mkdir(parents=True, exist_ok=True)
    # Return the path
    return default_root