# The Privacy Power of Correlated Noise in Decentralized Learning

## Requirements

* Download library requirements using `pip install -r requirements.txt`

## Paper figures:

* The results of experiments on Synthetic data can be found in the folder: [loss_epsilon](quadratics_for_n_16/loss_epsilon/)
* The plot on LibSVM can be found in:  [results-plot-libsvm](results-plot-libsvm/)
* The plot on MNIST can be found in:  [results-plot-mnist](results-plot-mnist/)
  
## Reproducing the paper's experiments:

* Run file [quadratics.py](quadratics.py) or command `python3 quadratics.py` to get our experiments on Synthetic data (Linear Regression).
* Run file [reproduce_libsvm.py](reproduce_libsvm.py) or command `python3 reproduce_libsvm.py` to get the experiements on the LibSVM dataset [a9a](libsvm_data/). You can also add to the previous command `--supercharge x` to run 2x experiments at the same time. Numerical results will be shown in the folder: [results-data-libsvm](results-data-libsvm/) and the plots in: [results-plot-libsvm](results-plot-libsvm/)

* Run file [reproduce_mnist.py](reproduce_mnist.py) or command `python3 reproduce_mnist.py` to get the experiements on the MNIST dataset (available on torchvision library in PyTorch). You can also add to the previous command `--supercharge x` to run 2x experiments at the same time. Numerical results will be shown in the folder: [results-data-mnist](results-data-mnist/) and the plots in: [results-plot-mnist](results-plot-mnist/)

* Run file [tuning_libsvm.py](tuning_libsvm.py) to tune hyperparameters for LibSVM experiments, and [tuning_mnist.py](tuning_mnist.py) for MNIST. Use the lists: `lr_grid` and `gradient_clip_grid` to set the values of learning rate and clipping to use in the grid search.
 
