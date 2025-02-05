import numpy as np

def linear_kernel(x1, x2, **kwargs):
    return np.dot(x1, x2)

def polynomial_kernel(x1, x2, degree=3, coef0=1, gamma=1, **kwargs):
    return (gamma * np.dot(x1, x2) + coef0) ** degree

def rbf_kernel(x1, x2, gamma=0.1, **kwargs):
    return np.exp(-gamma * np.linalg.norm(x1 - x2)**2)
