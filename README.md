# The Privacy Power of Correlated Noise in Decentralized Learning

This is the official code repository of the paper: The Privacy Power of Correlated Noise in Decentralized Learning.

## Abstract

Decentralized learning is appealing as it enables the scalable usage of large amounts of distributed data and resources (without resorting to any central entity), while promoting privacy since every user minimizes the direct exposure of their data. Yet,  without additional precautions, curious users can still leverage models obtained from their peers to violate privacy. Here, we propose DECOR, a variant of decentralized SGD with differential privacy (DP) guarantees. In DECOR, users securely exchange randomness seeds in one communication round to generate pairwise-canceling correlated Gaussian noises, which are injected to protect local models at every communication round. We theoretically and empirically show that, for arbitrary connected graphs, DECOR matches the central DP optimal privacy-utility trade-off. We do so under SecLDP, our new relaxation of local DP, which protects all user communications against an external eavesdropper and curious users, assuming that every pair of connected users shares a secret, i.e., an information hidden to all others. The main theoretical challenge is to control the accumulation of non-canceling correlated noise due to network sparsity. We also propose a companion SecLDP privacy accountant for public use.

## Requirements

* Download library requirements using `pip install -r requirements.txt`


## Paper figures:

* The results of experiments on Synthetic data can be found in the folder: [loss_epsilon](quadratics_for_n_16/loss_epsilon/)
* The plot on LibSVM can be found in:  [results-plot-libsvm](results-plot-libsvm/)
* The plot on MNIST can be found in:  [results-plot-mnist](results-plot-mnist/)
  
## Reproducing the paper's experiments:

* Run file [quadratics.py](quadratics.py) or command `python3 quadratics.py` to get our experiments on Synthetic data (Linear Regression).
* Run file [reproduce_libsvm.py](reproduce_libsvm.py) or command `python3 reproduce_libsvm.py` to get the experiements on the LibSVM dataset [a9a](libsvm_data/). You can also add to the previous command `--supercharge x` to run 2x experiments at the same time. Numerical results will be shown in the folder: `results-data-libsvm` and the plots in: [results-plot-libsvm](results-plot-libsvm/)

* Run file [reproduce_mnist.py](reproduce_mnist.py) or command `python3 reproduce_mnist.py` to get the experiements on the MNIST dataset (available on torchvision library in PyTorch). You can also add to the previous command `--supercharge x` to run 2x experiments at the same time. Numerical results will be shown in the folder: `results-data-libsvm` and the plots in: [results-plot-mnist](results-plot-mnist/)

* Run file [tuning_libsvm.py](tuning_libsvm.py) to tune hyperparameters for LibSVM experiments, and [tuning_mnist.py](tuning_mnist.py) for MNIST. Use the lists: `lr_grid` and `gradient_clip_grid` to set the values of learning rate and clipping to use in the grid search.
 
