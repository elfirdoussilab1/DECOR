# The Privacy Power of Correlated Noise in Decentralized Learning

## Instructions

* Download library requirements using `pip install -r requirements.txt`
### Reproducing the paper's experiments:
* Run file [quadratics.py](quadratics.py) or command `python3 quadratics.py` to get our experiments on Synthetic data (Linear Regression).
* Run file [reproduce_libsvm.py](reproduce_libsvm.py) or command `python3 reproduce_libsvm.py` to get the experiements on the LibSVM dataset [a9a](libsvm_data/). You can also add to the previous command `--supercharge x` to run 2x experiments at the same time. Numerical results will be shown in the folder: [results-data-libsvm/](results-data-libsvm/) and the plots in: [results-plot-libsvm/](results-plot-libsvm/)
* Run file 'reproduce_mnist.py' or command `python3 reproduce_mnist.py` to get the experiements on the MNIST dataset (available on torchvision library in PyTorch).
* Fetch results in folder XXX

## Reference
If you use this code, please cite the following paper

```
@inproceedings{vogels2021relaysum,
  title={The Privacy Power of Correlated Noise in Decentralized Learning},
  author={Youssef Allouah, Anastasia Koloskova, Aymane El Firdoussi, Rachid Guerraoui and Martin Jaggi},
  booktitle={The Forty-first International Conference on Machine Learning},
  year={2024}
}
```
