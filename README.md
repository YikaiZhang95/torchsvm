# torchsvm
![img](https://release-badges-generator.vercel.app/api/releases.svg?user=YikaiZhang95&repo=torchsvm&gradient=0000ff,8bd1fa)

This is a PyTorch-based package to solve kernel SVM with GPU.

## Table of contents

* [Introduction](#introduction)
* [Installation](#installation)
* [Quick start](#quick-start)
* [Usage](#usage)
* [Getting help](#getting-help)

## Introduction

torchsvm, a PyTorch-based library that trains kernel SVMs and other large-margin classifiers with exact leave-one-out cross-validation (LOOCV) error computation. Conventional SVM solvers often face scalability and efficiency challenges, especially on large datasets or when multiple cross-validation runs are required. torchsvm computes LOOCV at the same cost as training a single SVM while boosting speed and scalability via CUDA-accelerated matrix operations. Benchmark experiments indicate that TorchKSVM outperforms existing kernel SVM solvers in efficiency and speed. 


## Installation

You can use `pip` to install this package.

```sh
pip install torchsvm
```


## Quick start

The usages are similar with `scikit-learn`:

```python
model = cvksvm(Kmat=Kmat, y=y_train, nlam=nlam, ulam=ulam, foldid=foldid, nfolds=nfolds, eps=1e-5, maxit=1000, gamma=1e-8, is_exact=0, device='cuda')
model.fit()
```

## Usage

### Generate simulation data
`torchsvm` provides a simulation data generation function to test functions in the library:

```python
# Sample data
nn = 10000 # Number of samples
nm = 5    # Number of clusters per class
pp = 10   # Number of features
p1 = p2 = pp // 2    # Number of positive/negative centers
mu = 2.0  # Mean shift
ro = 3  # Standard deviation for normal distribution
sdn = 42  # Seed for reproducibility

nlam = 50
torch.manual_seed(sdn)
ulam = torch.logspace(3, -3, steps=nlam)

X_train, y_train, means_train = data_gen(nn, nm, pp, p1, p2, mu, ro, sdn)
X_test, y_test, means_test = data_gen(nn // 10, nm, pp, p1, p2, mu, ro, sdn)
X_train = standardize(X_train)
X_test = standardize(X_test)

sig = sigest(X_train)
Kmat = rbf_kernel(X_train, sig)
```


### Basic operation

`torchsvm` mainly provides `cvksvm` to tune kernel SVM fast with GPU acceleration and compute exact leave-one-out cross-validation (LOOCV) errors if needed.

```python
model = cvksvm(Kmat=Kmat, y=y_train, nlam=nlam, ulam=ulam, foldid=foldid, nfolds=nfolds, eps=1e-5, maxit=1000, gamma=1e-8, is_exact=0, device='cuda')
model.fit()
```
It also provides applications for other large-margin classifiers:

1. Kernel logistic regression
   ```python
    model = cvklogit(Kmat=Kmat, y=y_train, nlam=nlam, ulam=ulam, foldid=foldid, nfolds=nfolds, eps=1e-5, maxit=1000, gamma=1e-8, is_exact=0, device='cuda')
    model.fit()
    ```
2. Kernel SVM with Huber loss
    ```python
    model = cvkhuber(Kmat=Kmat, y=y_train, nlam=nlam, ulam=ulam, foldid=foldid, nfolds=nfolds, eps=1e-5, maxit=1000, gamma=1e-8, is_exact=0, device='cuda')
    model.fit()
    ```
3. Kernel squared SVM
   ```python
    model = cvksqsvm(Kmat=Kmat, y=y_train, nlam=nlam, ulam=ulam, foldid=foldid, nfolds=nfolds, eps=1e-5, maxit=1000, gamma=1e-8, is_exact=0, device='cuda')
    model.fit()
    ```  


## Getting help

Any questions or suggestions please contact: <yikai-zhang@uiowa.edu>


