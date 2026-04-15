from discreteNPIV.utils import * 
import numpy as np
from sklearn.model_selection import KFold
 
 



def compute_terms_primal_2SLS(X, Z, Y):
    """
    Solves the primal npJIVE (nonparametric Joint Instrumental Variable Estimator) problem.
    Instead of computing theta_hat multiple times, it averages M_reg and b across repeats and solves once.

    Args:
        X (numpy.ndarray): Feature matrix of shape (n, d).
        Z (numpy.ndarray): Instrument variable, categorical array of shape (n,).
        Y (numpy.ndarray): Response variable of shape (n,).
        K (int): Number of instrument categories.
        cross_folds (list or numpy.ndarray, optional): List of cross-validation folds or a single fold array.
        lambda_K (float, optional): Regularization parameter for the estimator (default: 1e-3).
        num_repeats (int, optional): Number of times to regenerate cross_folds and recompute (default: 3).
        gamma (float, optional): Ridge regularization parameter (default: 0).

    Returns:
        tuple:
            - theta_hat (numpy.ndarray): Estimated parameter vector of shape (d,).
            - struct_fun (function): Function that takes X as input and returns predictions X @ theta_hat.
    """
    K = len(set(Z))
    n_total, d = X.shape
    group_counts = np.bincount(Z)
    W = group_counts / n_total  # shape (K,)
    # Compute T_X_basis and T_Y (if provided)
    T_X = compute_group_means(X, Z, K)
    T_Y = compute_group_means(Y, Z, K)
    M = (T_X.T * W) @ T_X
    b = (T_X.T * W) @ T_Y
    return M, b





def solve_primal_structural_2SLS(X, Z, Y, lambda_K=1e-3,  gamma=0):
    """
    Solves the primal npJIVE (nonparametric Joint Instrumental Variable Estimator) problem.
    Instead of computing theta_hat multiple times, it averages M_reg and b across repeats and solves once.

    Args:
        X (numpy.ndarray): Feature matrix of shape (n, d).
        Z (numpy.ndarray): Instrument variable, categorical array of shape (n,).
        Y (numpy.ndarray): Response variable of shape (n,).
        K (int): Number of instrument categories.
        cross_folds (list or numpy.ndarray, optional): List of cross-validation folds or a single fold array.
        lambda_K (float, optional): Regularization parameter for the estimator (default: 1e-3).
        num_repeats (int, optional): Number of times to regenerate cross_folds and recompute (default: 3).
        gamma (float, optional): Ridge regularization parameter (default: 0).

    Returns:
        tuple:
            - theta_hat (numpy.ndarray): Estimated parameter vector of shape (d,).
            - struct_fun (function): Function that takes X as input and returns predictions X @ theta_hat.
    """
    K = len(set(Z))
    n_total, d = X.shape
    M, b = compute_terms_primal_2SLS(X, Z, Y)
    S = X.T @ X / n_total
    M_reg = M + lambda_K * S + gamma  * np.eye(d)
    theta_hat = lstsq_fast(M_reg, b)
    return theta_hat


def compute_primal_risk_2SLS(preds, Z, X, Y, K, lambda_min=1e-5):
    n_total = len(Z)
    group_counts = np.bincount(Z, minlength=K)
    W = group_counts / n_total  # Group weights
    
    err_group = compute_group_means(Y - preds, Z, K)

    risk = np.sum(W * err_group * err_group)  
    risk += lambda_min * np.mean(preds**2)
    return risk

 





def compute_terms_dual_2SLS(X, Z, X_new = None):
    """
    Solves the primal npJIVE (nonparametric Joint Instrumental Variable Estimator) problem.
    Instead of computing theta_hat multiple times, it averages M_reg and b across repeats and solves once.

    Args:
        X (numpy.ndarray): Feature matrix of shape (n, d).
        Z (numpy.ndarray): Instrument variable, categorical array of shape (n,).
        Y (numpy.ndarray): Response variable of shape (n,).
        K (int): Number of instrument categories.
        cross_folds (list or numpy.ndarray, optional): List of cross-validation folds or a single fold array.
        lambda_K (float, optional): Regularization parameter for the estimator (default: 1e-3).
        num_repeats (int, optional): Number of times to regenerate cross_folds and recompute (default: 3).
        gamma (float, optional): Ridge regularization parameter (default: 0).

    Returns:
        tuple:
            - theta_hat (numpy.ndarray): Estimated parameter vector of shape (d,).
            - struct_fun (function): Function that takes X as input and returns predictions X @ theta_hat.
    """
    K = len(set(Z))
    n_total, d = X.shape
    group_counts = np.bincount(Z)
    W = group_counts / n_total  # shape (K,)
    # Compute T_X_basis and T_Y (if provided)
    T_X = compute_group_means(X, Z, K)
    M = (T_X.T * W) @ T_X
    if X_new is not None:
        b = np.mean(X_new, axis=0)
    else:
        b = None
    return M, b



def solve_dual_riesz_2SLS(X, Z, X_new, lambda_K=1e-3, gamma=0):
    """
    Solves the dual problem for the Riesz representer with Thikokov and ridge regularization.

    Args:
        X (numpy.ndarray): Feature matrix of shape (n, d).
        Z (numpy.ndarray): Instrument variable, categorical array of shape (n,).
        K (int): Number of instrument categories.
        X_new (numpy.ndarray, optional): New feature matrix for out-of-sample predictions (default: None, uses X).
        lambda_K (float, optional): Regularization parameter for the estimator (default: 1e-3).
        gamma (float, optional): Ridge regularization parameter (default: 0).

    Returns:
        tuple:
            - b_hat (numpy.ndarray): Estimated dual solution of shape (d,).
            - beta_hat (numpy.ndarray): Estimated function evaluation X @ b_hat.
    """
    
    K = len(set(Z))
    n_total, d = X.shape
    M, b = compute_terms_dual_2SLS(X, Z, X_new)
    S = X.T @ X / n_total
    M_reg = M + lambda_K * S + gamma  * np.eye(d)
    theta_hat = lstsq_fast(M_reg, b)
    return theta_hat





def compute_dual_risk_2SLS(preds, preds_new, Z, K, lambda_min=1e-4):
    """
    Computes the dual risk given the estimated solution b_hat.
 

    Args:
        preds (numpy.ndarray): Predicted values for training data, shape (n,).
        preds_new (numpy.ndarray): Predicted values for out-of-sample validation, shape (n,).
        Z (numpy.ndarray): Instrument variable, categorical array of shape (n,).
        K (int): Number of instrument categories.
        lambda_min (float, optional): Regularization term to prevent division issues (default: 1e-4).

    Returns:
        float: Averaged computed dual risk value.
    """
    n_total = len(Z)
    group_counts = np.bincount(Z, minlength=K)
    W = group_counts / n_total  # Group weights
    dual_group = compute_group_means(preds, Z, K)
    dual_risk = np.sum(W * dual_group ** 2) - 2 * np.mean(preds_new)
    return dual_risk
    
