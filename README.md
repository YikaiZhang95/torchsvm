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

`torchsvm`, a PyTorch-based library that trains kernel SVMs and other large-margin classifiers with exact leave-one-out cross-validation (LOOCV) error computation. Conventional SVM solvers often face scalability and efficiency challenges, especially on large datasets or when multiple cross-validation runs are required. torchsvm computes LOOCV at the same cost as training a single SVM while boosting speed and scalability via CUDA-accelerated matrix operations. Benchmark experiments indicate that TorchKSVM outperforms existing kernel SVM solvers in efficiency and speed. This document shows how to use the `torchsvm` package to fit kernel SVM.

When dealing with low-dimensional problems or more complex scenarios, such as requiring non-linear decision boundaries or higher accuracy, kernel SVMs can be formulated using the kernel method within a reproducing kernel Hilbert space (RKHS). For consistency, we adopt the same notation introduced in the high-dimensional case in Chapter One.

Given a random sample $\\{y_i, x_i\\}_{i=1}^n$, the kernel SVM can be formulated as a function estimation problem:

![kernel SVM formulation](https://latex.codecogs.com/svg.image?\dpi{130}&space;\min_{f&space;\in&space;\mathcal{H}_K}&space;\left[&space;\frac{1}{n}&space;\sum_{i=1}^n&space;\left(&space;1&space;-&space;y_i&space;f(\mathbf{x}_i)&space;\right)_{+}&space;&plus;&space;\lambda&space;\|f\|_{\mathcal{H}_K}^2&space;\right])

where ![norm](https://latex.codecogs.com/svg.image?\dpi{120}&space;\left\|f\right\|^2_{\mathcal{H}_K}) is the RKHS norm that acts as a regularizer, and $\lambda > 0$ is a tuning parameter.

According to the representer theorem for reproducing kernels (Wahba, 1990), the solution to our problem takes the form:

![f(x) formula](https://latex.codecogs.com/svg.image?\dpi{130}&space;f(\mathbf{x})&space;=&space;\sum_{i=1}^n&space;\alpha_i^{\mathrm{SVM}}&space;K\left(\mathbf{x}_i,&space;\mathbf{x}\right))

The coefficients $\alpha^{SVM}$ are obtained by solving the optimization problem:

![alpha optimization](https://latex.codecogs.com/svg.image?\dpi{130}&space;\boldsymbol{\alpha}^{\mathrm{SVM}}&space;=&space;\arg\min_{\boldsymbol{\alpha}&space;\in&space;\mathbb{R}^n}&space;\left[&space;\frac{1}{n}&space;\sum_{i=1}^n&space;\left(1&space;-&space;y_i&space;\mathbf{K}_i^{\top}&space;\boldsymbol{\alpha}&space;\right)_{+}&space;&plus;&space;\lambda&space;\boldsymbol{\alpha}^\top&space;\mathbf{K}&space;\boldsymbol{\alpha}&space;\right])

where $\mathbf{K}$ is the kernel matrix.


## Installation

You can use `pip` to install this package.

```sh
pip install torchsvm
```


## Quick start
Import necessary libraries and functions:

```python
from torchsvm.cvksvm import cvksvm
from torchsvm.functions import *
import torch
import numpy
```

The usages are similar with `scikit-learn`:

```python
model = cvksvm(Kmat=Kmat, y=y_train, nlam=nlam, ulam=ulam, nfolds=nfolds, eps=1e-5, maxit=1000, gamma=1e-8, is_exact=0, device='cuda')
model.fit()
```

## Usage

### Generate simulation data
`functions` provides a simulation data generation function, `data_gen`, to generate data from a mixture of Gaussian models. `functions` also provides kernel operations like `rbf_kernel` and `kernelMult`, as well as data processing functions such as `standardize`.

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
model = cvksvm(Kmat=Kmat, y=y_train, nlam=nlam, ulam=ulam, nfolds=5, eps=1e-5, maxit=1000, gamma=1e-8, is_exact=0, device='cuda')
model.fit()
```

```python
### Tune parameter
cv_mis = model.cv(model.pred, y_train).numpy()
best_ind = numpy.argmin(cv_mis)
best_ind

### Test Error and objective value
Kmat = Kmat.double()
alpmat = model.alpmat.to('cpu')
intcpt = alpmat[0,best_ind]
alp = alpmat[1:,best_ind]
ka = torch.mv(Kmat, alp)
aka = torch.dot(alp, ka)
obj_magic = model.objfun(intcpt, aka, ka, y_train, ulam[best_ind], nn)

Kmat_new = kernelMult(X_test, X_train, sig)
Kmat_new = Kmat_new.double()

result = torch.mv(Kmat_new, alpmat[1:,best_ind]) + alpmat[0, best_ind]

ypred = torch.where(result > 0, torch.tensor(1), torch.tensor(-1))

torch.mean((ypred == y_test).float())
```

### Real data

`torchsvm` works well with `sklearn` datasets. We need to convert these datasets to `torch.tensor` with $y=1 \text{ or} -1$.

```python 
from sklearn.datasets import make_moons

# Generate non-linear dataset
X, y = make_moons(n_samples=300, noise=0.2, random_state=42)

X = torch.from_numpy(X).float()
y = torch.from_numpy(y).float()
y = 2 * y - 1

sig = sigest(X)
Kmat = rbf_kernel(X, sig)

model = cvksvm(Kmat=Kmat, y=y, nlam=nlam, ulam=ulam, nfolds=nfolds, eps=1e-5, maxit=1000, gamma=1e-8, is_exact=0, device='cuda')
model.fit()
```

### Extensions to large-margin classifiers 
It also provides applications for other large-margin classifiers:

1. Kernel logistic regression
   ```python
    model = cvklogit(Kmat=Kmat, y=y_train, nlam=nlam, ulam=ulam, foldid=foldid, nfolds=nfolds, eps=1e-5, maxit=1000, gamma=1e-8, is_exact=0, device='cuda')
    model.fit()
    ```
2. Kernel distance weighted discrimination
    ```python
    model = cvkdwd(Kmat=Kmat, y=y_train, nlam=nlam, ulam=ulam, foldid=foldid, nfolds=nfolds, eps=1e-5, maxit=1000, gamma=1e-8, is_exact=0, device='cuda')
    model.fit()
    ```


## Getting help

Any questions or suggestions please contact: <yikai-zhang@uiowa.edu>


