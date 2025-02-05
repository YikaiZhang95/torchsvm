import torch

def data_gen(nn, nm, pp, p1, p2, mu, ro, sdn=None, means=None):
    """
    Generate synthetic data with positive and negative centers.
    
    Parameters:
    - nn (int): Number of samples (total observations).
    - nm (int): Number of clusters (per class).
    - pp (int): Number of features.
    - p1 (int): First set of positive centers.
    - p2 (int): Second set of positive centers (unused in original code).
    - mu (float): Mean shift for positive and negative centers.
    - ro (float): Standard deviation for normal distribution.
    - sdn (int, optional): Seed for reproducibility.
    - means (torch.Tensor, optional): Predefined cluster means (if any).
    
    Returns:
    - X (torch.Tensor): Feature matrix.
    - y (torch.Tensor): Labels vector.
    - means (torch.Tensor): Cluster centers.
    """
    
    # Set seed if provided
    if sdn is not None and means is None:
        torch.manual_seed(sdn)
        means = torch.randn(nm * 2, pp)
        # Negative centers: Shift the first `p1` features
        means[:nm, :p1] += mu
        # Positive centers: Shift the remaining features
        means[nm:, p1:pp] += mu

    # Generate binary labels (randomly assign 1 and -1)
    id_pos = torch.bernoulli(torch.full((nn,), 0.5)).bool()
    size_pos = torch.sum(id_pos).item()
    size_neg = nn - size_pos

    # Initialize labels
    y = torch.full((nn,), -1.0)
    y[id_pos] = 1.0

    # Generate random features from normal distribution
    X = torch.randn(nn, pp) * ro

    # Assign random cluster IDs for negative and positive samples
    ids = torch.empty(nn).long()
    ids[~id_pos] = torch.randint(0, nm, (size_neg,))
    ids[id_pos] = torch.randint(nm, nm * 2, (size_pos,))

    # Adjust features based on the cluster centers
    X += means[ids]

    return X, y, means

def sigest(x, frac=0.5):
    """
    PyTorch equivalent of the R function sigest.
    
    Parameters:
    - x (torch.Tensor): Input tensor of shape (m, n), where m is the number of samples and n is the number of features.
    - frac (float): Fraction of samples to use for computing the distance.
    
    Returns:
    - sigma_estimate (float): Estimated sigma based on quantiles of squared distances.
    """
    
    # Number of samples (m)
    m = x.shape[0]
    
    # Number of random samples to take for the distance calculation
    n = int(frac * m)
    
    # Randomly sample `n` indices (two sets)
    index1 = torch.randint(0, m, (n,), dtype=torch.long)
    index2 = torch.randint(0, m, (n,), dtype=torch.long)
    
    # Compute the squared differences between the randomly paired rows
    temp = x[index1] - x[index2]
    dist = torch.sum(temp ** 2, dim=1)
    
    # Exclude zero distances (self-pairs)
    non_zero_dist = dist[dist != 0]
    
    # Compute quantiles (0.9, 0.5, 0.1)
    srange = 1.0 / torch.quantile(non_zero_dist, torch.tensor([0.9, 0.5, 0.1]))
    
    # Return the mean of the 90th and 10th quantiles
    sigma_estimate = torch.mean(srange[[0, 2]]).item()

    return sigma_estimate

def rbf_kernel(x, sigma):
    """
    Compute the RBF (Gaussian) kernel matrix in PyTorch.

    Parameters:
    - x (torch.Tensor): Input tensor of shape (n_samples, n_features).
    - sigma (float): The standard deviation parameter for the RBF kernel (Gaussian width).

    Returns:
    - K (torch.Tensor): RBF kernel matrix of shape (n_samples, n_samples).
    """
    # Compute pairwise squared Euclidean distances
    x_norm = torch.sum(x ** 2, dim=1).view(-1, 1)
    pairwise_dists = x_norm + x_norm.t() - 2.0 * torch.mm(x, x.t())

    # Compute the RBF kernel matrix
    K = torch.exp(-pairwise_dists * 2 * sigma)

    return K

def standardize(x):
    """
    Standardizes the input tensor (feature-wise standardization).
    
    Args:
    - x (torch.Tensor): Input tensor (matrix) of shape (n_samples, n_features).
    
    Returns:
    - x_standardized (torch.Tensor): Standardized tensor where each feature has mean 0 and standard deviation 1.
    """
    # Compute column-wise means and standard deviations
    mean = torch.mean(x, dim=0)
    std = torch.std(x, dim=0)

    # Replace zeros in std with 1 to avoid division by zero
    std[std == 0] = 1

    # Standardize: subtract the mean and divide by the standard deviation
    x_standardized = (x - mean) / std
    return x_standardized

def kernelMult(X, X_new, sigma):
    """
    Compute the RBF (Gaussian) kernel matrix between X and X_new in PyTorch.

    Parameters:
    - X (torch.Tensor): Input tensor of shape (n_samples_X, n_features).
    - X_new (torch.Tensor): Input tensor of shape (n_samples_X_new, n_features).
    - sigma (float): The standard deviation parameter for the RBF kernel (Gaussian width).

    Returns:
    - K (torch.Tensor): RBF kernel matrix of shape (n_samples_X, n_samples_X_new).
    """
    # Compute squared L2 norms
    X_norm = torch.sum(X ** 2, dim=1).view(-1, 1)
    X_new_norm = torch.sum(X_new ** 2, dim=1).view(1, -1)

    # Compute pairwise squared Euclidean distances
    pairwise_dists = X_norm + X_new_norm - 2.0 * torch.mm(X, X_new.t())

    # Compute the RBF kernel matrix
    K = torch.exp(-pairwise_dists * 2 * sigma)

    return K