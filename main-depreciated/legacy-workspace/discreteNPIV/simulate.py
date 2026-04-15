import numpy as np

def generate_data(n, K, d=5, n_new=10000, sigma_UX=0.1, sigma_UY=0.1, sigma_UC=0.1, sparsity1=None, sparsity2=None, seed_setting=42):

    if sparsity1 is None:
        sparsity1 = min(5, 0.1 * d)
    if sparsity2 is None:
        sparsity2 = sparsity1 
    alpha1 = find_alpha(sparsity1, d)
    alpha2 = find_alpha(sparsity2, d)

    
    decay = np.array([j**(-alpha2) for j in range(1, d + 1)])  # Sobolev-type decay
    n_new = int(n_new)
    rng_fixed = np.random.default_rng(100000 * seed_setting)  # Create a generator with a fixed seed
    Pi_new = rng_fixed.standard_normal(d) * decay  # Shape (d,)
    X_new = Pi_new[np.newaxis, :].repeat(n_new, axis=0) + rng_fixed.standard_normal((n_new, 1)) * sigma_UX + rng_fixed.standard_normal((n_new, 1)) * sigma_UC

    
    # Generate Pi matrix with Gaussian decay following N(0, 1/K)
    rng_fixed = np.random.default_rng(seed_setting)
    Pi = rng_fixed.standard_normal((K,d)) * decay[np.newaxis, :]  # Shape (K, d)

    # Construct theta
    decay = np.array([j**(-alpha1) for j in range(1, d + 1)])
    rng_fixed = np.random.default_rng(seed_setting)
    theta = (2 * (rng_fixed.standard_normal(d) >= 0) - 1) * decay  # Shape (d,)
 

    # Generate independent noise terms for each (k, i)
    rng = np.random.default_rng()
    U_X = rng.standard_normal((n, K)) * sigma_UX  # Shape (n, K)
    U_Y = rng.standard_normal((n, K)) * sigma_UY  # Shape (n, K)
    U_C = rng.standard_normal((n, K)) * sigma_UC  # Shape (n, K)

    # Compute X_{ki} ensuring dependence on U_X, U_C, and Pi
    X = Pi[None, :, :] + U_C[:, :, None] + U_X[:, :, None]  # Shape (n, K, d)
    X = X.reshape(n * K, d)  # Reshape to (n*K, d)

    # Compute X_{ki}^T theta
    X_theta = X @ theta  # Shape (n*K,)

    # Generate Y_{ki} using independent U_Y and U_C
    Y = X_theta + U_Y.ravel() + U_C.ravel()  # Shape (n*K,)

    # Generate one-hot encoding for category K
    Z = np.tile(np.arange(K), n)  # Shape (n*K,)

    return {
        "theta": theta,  # Shape (d,)
        "Y": Y,  # Shape (n*K,)
        "X": X,  # Shape (n*K, d)
        "Z": Z,  # Shape (n*K,)
        "Pi": Pi,  # Shape (K, d)
        "X_new": X_new,  # Shape (n_new, d),
        "alpha1": alpha1,
        "alpha2":alpha2
    }


import numpy as np
from scipy.optimize import bisect

def effective_sparsity(alpha, j_vals):
    """
    Compute the effective sparsity for a Sobolev-type decay vector v_j = j^(-alpha).
    """
    decay = j_vals ** (-alpha)
    sum1 = np.sum(decay)
    sum2 = np.sum(decay**2)
    return (sum1 ** 2) / sum2

def find_alpha(target_sparsity, d, tol=1e-6):
    """
    Find the alpha such that the effective sparsity matches the target.
    """
    j_vals = np.arange(1, d + 1)

    def objective(alpha):
        return effective_sparsity(alpha, j_vals) - target_sparsity

    # Use bisection method for a robust solution
    alpha = bisect(objective, 0, 10, xtol=tol)
    return alpha

def truncation_error(decay, s):
    """
    Compute the relative l^2 error of truncating decay vector to its first s components.
    """
    total_energy = np.sum(decay**2)
    truncated_energy = np.sum(decay[:s]**2)
    return 1 - (truncated_energy / total_energy)



def compute_metrics(theta_hat, theta_true, Z, X, Y, folds, K):
    n_total = len(Z)
    group_counts = np.bincount(Z, minlength=K)
    W = group_counts / n_total
    res = X @ theta_true - X @ theta_hat
    res_1 = compute_group_means(res, Z, folds, 1, K)
    res_0 = compute_group_means(res, Z, folds, 0, K)
    
    strong_norm = np.sqrt(np.mean(res**2))
    weak_norm = np.sqrt(np.mean(res_1 * res_0))
    jive_risk = np.mean(W * compute_group_means(Y - X @ theta_hat, Z, folds, 1, K) * compute_group_means(Y - X @ theta_hat, Z, folds, 0, K)) / np.mean(W)
    
    print(f"{'Metric':<15}{'Value'}")
    print("=" * 30)
    print(f"{'Strong Norm':<15}{strong_norm:.6f}")
    print(f"{'Weak Norm':<15}{weak_norm:.6f}")
    print(f"{'JIVE Risk':<15}{jive_risk:.6f}")
    
    return 