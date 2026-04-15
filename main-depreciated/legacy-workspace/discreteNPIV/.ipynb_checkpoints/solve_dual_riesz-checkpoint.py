from discreteNPIV.utils import * 
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
import time
from discreteNPIV.npiv_2SLS import * 
 
 
def npiv_dual(X, Z, X_new, num_lambda=20, num_gamma=10, 
                lambda_grid=None, gamma_grid=None, n_splits=10,adaptive = True, verbose=False):
    """
    Solve the dual inverse problem for estimation of the Riesz representer of the linear functional in the discrete Nonparametric Instrumental Variable (NPIV) problem.


    Args:
        X (numpy.ndarray): Feature matrix of shape (n, d).
        Z (numpy.ndarray): Instrumental variable, categorical array of shape (n,).
        K (int): Number of instrument categories.
        X_new : ndarray of shape (m_samples, d)
            New set of covariates for computing the mean of the structural function (defines the estimand).
        cross_folds (list or numpy.ndarray, optional): Predefined cross-validation folds.
        num_lambda (int, optional): Number of lambda values to search in cross-validation (default: 20).
        num_gamma (int, optional): Number of gamma values to search in cross-validation (default: 10).
        lambda_grid (numpy.ndarray, optional): Grid of lambda values for regularization (default: None).
        gamma_grid (numpy.ndarray, optional): Grid of gamma values for regularization (default: None).
        n_splits (int, optional): Number of cross-validation splits (default: 10).
        num_repeats (int, optional): Number of repetitions when solving npJIVE (default: 1).
        verbose (bool, optional): If True, prints progress and debugging information (default: False).

    Returns:
        tuple:
            - theta_hat (numpy.ndarray): Estimated parameter vector of shape (d,).
            - struct_fun (function): Function that takes X as input and returns predictions X @ theta_hat.
            - cv_output (tuple): Output from cross-validation, containing selected parameters and fold structure.
    """
    
    # Perform cross-validation to select the best regularization parameters
    cv_output = cross_validate_dual(X, Z, X_new, 
                                    num_lambda=num_lambda, num_gamma=num_gamma, 
                                  lambda_grid=lambda_grid, 
                                    gamma_grid=gamma_grid, n_splits=n_splits,  verbose=verbose)

    params_jive, params_2sls, use_npjiv = cv_output

    # Solve the estimation problem using either npJIVE or a standard 2SLS approach
    b_hat_jive = solve_dual_riesz(X, Z, X_new,
                                               lambda_K=params_jive['lambda'],  
                                               gamma=params_jive['gamma'])
    b_hat_2sls = solve_dual_riesz_2SLS(X, Z, X_new,
                                                 lambda_K=params_2sls['lambda'],  
                                                 gamma=params_2sls['gamma'])

    if use_npjiv or not adaptive:
        b_hat = b_hat_jive
    else:
        b_hat = b_hat_2sls
    fit_info = {"jive": b_hat_jive, "2sls": b_hat_2sls, "cv_output": cv_output}
    return b_hat, fit_info
 

def compute_terms_dual_jive(X, Z, X_new):
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
    n_total, d = X.shape
    # Compute T_X_basis and T_Y (if provided)
    T_X = compute_means_jackknife(X, Z)
    # Precompute transposes
    X_t = X.T
    T_X_t = T_X.T
    # Precompute products for matrix M
    TX_X = T_X_t @ X / n_total  # T_X_basis^T X_basis
    X_TX = X_t @ T_X / n_total  # X_basis^T T_X_basis
    M = (TX_X + X_TX)/2  # Matrix M
    # Output
    b = np.mean(X_new, axis=0)
    return M, b



def solve_dual_riesz(X, Z, X_new, lambda_K=1e-3,  gamma=0):
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
    n_total, d = X.shape
    M, b = compute_terms_dual_jive(X, Z, X_new)
    S = X.T @ X / n_total
    M_reg = M + lambda_K * S + gamma  * np.eye(d)
    theta_hat = lstsq_fast(M_reg, b)
    return theta_hat




 

import numpy as np
from sklearn.model_selection import StratifiedKFold



def cross_validate_dual(
    X, Z, X_new, num_lambda=20, num_gamma=10, lambda_grid=None, gamma_grid=None, n_splits=10,verbose=False):
    """
    Performs joint K-fold cross-validation to determine the optimal regularization parameters 
    (lambda_K, gamma) for a dual-based estimator within the npJIVE framework. The function 
    also evaluates the Two-Stage Least Squares (2SLS) method as a baseline.

    The function systematically explores a descending sequence of regularization parameters, 
    leveraging early stopping to prevent overfitting and reduce computational complexity. 
    It precomputes validation splits and group means to enhance efficiency. The npJIVE estimator 
    is compared against 2SLS based on estimated risk, and instability checks ensure that artificially 
    low risk values do not lead to erroneous selections.

    Parameters
    ----------
    X : numpy.ndarray, shape (n_samples, n_features)
        Feature matrix containing predictor variables.
    
    Z : numpy.ndarray, shape (n_samples,)
        Group assignment vector indicating the group membership for each row in `X`.
    
    K : int
        Number of distinct groups in `Z`.
    
    X_new : numpy.ndarray, shape (n_new_samples, n_features)
        Out-of-sample feature matrix used to evaluate generalization risk.
    
    num_lambda : int, optional, default=20
        Number of lambda values to consider if `lambda_grid` is not provided.
    
    num_gamma : int, optional, default=10
        Number of gamma values to consider if `gamma_grid` is not provided.
    
    cross_folds : numpy.ndarray, optional, default=None
        Predefined cross-validation fold assignments. If `None`, folds are generated automatically.
    
    lambda_grid : numpy.ndarray, optional, default=None
        Custom grid of lambda values. If `None`, a log-spaced grid between `10⁻¹` and `10⁻⁸` is generated.
    
    gamma_grid : numpy.ndarray, optional, default=None
        Custom grid of gamma values. If `None`, a log-spaced grid between `10⁻¹` and `10⁻⁸` is generated.
    
    n_splits : int, optional, default=10
        Number of cross-validation folds.
    
    lambda_min : float, optional, default=1e-8
        Minimum lambda value for numerical stability.
    
    max_risk_decrease_factor : float, optional, default=2
        Controls early stopping in the lambda parameter search. If the decrease in risk (`current_decrease`) 
        is more than `max_risk_decrease_factor` times the decrease observed from the previous lambda 
        (`prev_decrease`), further reductions in lambda are halted. This prevents selecting overly 
        aggressive regularization values that may lead to instability.
    
    max_improvement_over_2SLS_factor : float, optional, default=5
        Prevents selecting npJIVE parameters that yield an unrealistically low risk compared to 2SLS.
        If npJIVE’s risk is negative and falls below `max_improvement_over_2SLS_factor` times the 2SLS risk, 
        npJIVE is considered unstable and excluded from selection.
    
    verbose : bool, optional, default=False
        If `True`, prints detailed progress updates including the current lambda, gamma, risk estimates, 
        and early stopping conditions.

    Returns
    -------
    list
        A list containing:
        
        - dict: Selected parameters and risk for npJIVE:
            - `"method"` : `"npJIVE"`
            - `"lambda"` : Selected lambda value
            - `"gamma"` : Selected gamma value
            - `"risk"` : Estimated risk
            
        - dict: Selected parameters and risk for 2SLS:
            - `"method"` : `"2SLS"`
            - `"lambda"` : Selected lambda value
            - `"gamma"` : Selected gamma value
            - `"risk"` : Estimated risk
            
        - bool: Indicator of whether npJIVE is preferred over 2SLS based on risk comparison.
        
        - list: The cross-validation fold assignments used.
    """

    K = len(np.unique(Z))

    
    # Initialize cross-validation folds if not provided
    num_repeats = 1
     

    
    # Define grids in **descending** order
    lambda_grid = np.logspace(-1, -12, num_lambda) if lambda_grid is None else np.sort(lambda_grid)[::-1]
    gamma_grid = np.logspace(-1, -12, num_gamma) if gamma_grid is None else np.sort(gamma_grid)[::-1]
    num_lambda = len(lambda_grid)
    num_gamma = len(gamma_grid)
    
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    kf_new = KFold(n_splits=n_splits, shuffle=True)
    n_total, d = X.shape
    group_counts = np.bincount(Z, minlength=K)
    W = group_counts / n_total  # shape (K,)
    
 
    # **Precompute Training/Validation Splits and Group Means**
    import time
    start = time.time()
    fold_data = []
    splits = kf.split(X, Z)
    for (train_idx, val_idx) in kf.split(X, Z):
        X_train, X_val = X[train_idx], X[val_idx]
        X_new_train, X_new_val = X_new[train_idx], X_new[val_idx]
        Z_train, Z_val = Z[train_idx], Z[val_idx]
        
         
    
        M_train, b_train = compute_terms_dual_jive(X_train, Z_train, X_new_train)
        M_val, b_val = compute_terms_dual_jive(X_val, Z_val, X_new_val)
        M_train_no_split, _ = compute_terms_dual_2SLS(X, Z)
        b_train_no_split = b_train
        #M_val_no_split, b_val_no_split = compute_terms_primal_2SLS(X_val, Z_val, Y_val)
        S = X.T @ X / len(X)
        I = np.eye(len(M_train))
        
 
        fold_data.append((train_idx, val_idx, M_train, b_train, M_val, b_val,  M_train_no_split, b_train_no_split, S, I))

    

    # Storage for risk values
    risks = np.full((num_gamma, num_lambda), np.inf)
    best_lambda_risks = np.full(num_gamma, np.inf)
    risks_no_split = np.full((num_gamma, num_lambda), np.inf)
    best_lambda_risks_no_split = np.full(num_gamma, np.inf)
    start1 = time.time()
    # Precompute S (same for all folds)
    S = X.T @ X / len(X)  # shape (d, d)
    for j, gamma in enumerate(gamma_grid):
        min_risk = np.inf
        last_valid_lambda = None
        last_valid_risk = np.inf
        break_split = False
        prev_decrease = 0
        last_quad_term = 0
        
        min_risk_no_split = np.inf
        last_valid_lambda_no_split = None
        last_valid_risk_no_split = np.inf
        break_no_split = False

        for i, lambda_K in enumerate(lambda_grid):
            preds = np.zeros(n_total)
            risk = 0
            risk_no_split = 0
            quad_term = 0
             
            for (train_idx, val_idx, M_train, b_train, M_val, b_val,  M_train_no_split, b_train_no_split, S, I) in fold_data:
                if not break_split:
                    M_reg = M_train + lambda_K * S + gamma * I  # Regularized matrix
                    theta_hat = lstsq_fast(M_reg, b_train)
                    quad_term_fold = theta_hat.T @ M_val @ theta_hat
                    quad_term += quad_term_fold
                    risk += quad_term_fold - 2 * np.dot(b_val, theta_hat)
                if not break_no_split:
                    M_reg_no_split = M_train_no_split + lambda_K * S + gamma * I  # Regularized matrix
                    theta_hat_no_split = lstsq_fast(M_reg_no_split, b_train_no_split)
                    risk_no_split += theta_hat_no_split.T @ M_val @ theta_hat_no_split - 2 * np.dot(b_val, theta_hat_no_split)

            #print(time.time() - start)
            risk /= n_splits
            risk_no_split /= n_splits
            quad_term /= n_splits

            if verbose:
                print(f"Gamma: {gamma:.2e}, Lambda: {lambda_K:.2e}, Risk: {risk:.6f}, Quad term: {quad_term:.6f},  break: {break_split:.6f}")
                print(f"Split: {risk:6f}, Split: {risk_no_split:6f}")

            # unstable if quadratic term is negative or if risk is much smaller than no split.
            current_decrease = np.abs(risk - last_valid_risk)
            unstable = (quad_term < - 1 / n_total) 
            # less regularization should increase quadratic term (typically)
            unstable = unstable or quad_term < 0.99 * last_quad_term
 
            
            if break_split or (risk > min_risk or unstable):
                break_split = True  # Early stopping on first risk increase
            else:
                prev_decrease = current_decrease
                last_valid_lambda = lambda_K
                last_valid_risk = risk
                min_risk = risk
                last_quad_term = quad_term
            if break_no_split or (risk_no_split > min_risk_no_split):
                break_no_split = True  # Early stopping on first risk increase
            else:
                last_valid_lambda_no_split = lambda_K
                last_valid_risk_no_split = risk_no_split
                min_risk_no_split = risk_no_split

            if break_no_split and break_split:
                break

        # Store last stable lambda's risk
        if last_valid_lambda is not None:
            risks[j, np.where(lambda_grid == last_valid_lambda)[0][0]] = last_valid_risk
            best_lambda_risks[j] = last_valid_risk  # Track for gamma early stopping
            
        if last_valid_lambda_no_split is not None:
            risks_no_split[j, np.where(lambda_grid == last_valid_lambda_no_split)[0][0]] = last_valid_risk_no_split
            best_lambda_risks_no_split[j] = last_valid_risk_no_split  # Track for gamma early stopping
 

    print("time to tune primal: ", time.time() - start1, ", time to cv split: ", start1 - start)
    min_gamma_idx = np.argmin(best_lambda_risks)
    best_gamma = gamma_grid[min_gamma_idx]
    best_lambda = lambda_grid[np.argmin(risks[min_gamma_idx, :])]
    cv_risk = np.min(risks[min_gamma_idx, :])

    min_gamma_idx_no_split = np.argmin(best_lambda_risks_no_split)
    best_gamma_no_split = gamma_grid[min_gamma_idx_no_split]
    best_lambda_no_split = lambda_grid[np.argmin(risks_no_split[min_gamma_idx_no_split, :])]
    cv_risk_no_split = np.min(risks_no_split[min_gamma_idx_no_split, :])
    
    use_npjiv = cv_risk <= cv_risk_no_split
     


    if verbose:
        print(f"\nSelected Parameters and Risks:")
        print(f"{'-'*40}")
        print(f"npJIVE:")
        print(f"  Lambda: {best_lambda:.2e}")
        print(f"  Gamma:  {best_gamma:.2e}")
        print(f"  Risk:   {cv_risk:.8f}")
        print(f"{'-'*40}")
        print(f"2SLS:")
        print(f"  Lambda: {best_lambda_no_split:.2e}")
        print(f"  Gamma:  {best_gamma_no_split:.2e}")
        print(f"  Risk:   {cv_risk_no_split:.8f}")
        print(f"{'-'*40}")
        print(f"Using npJIVE: {'Yes' if use_npjiv else 'No'}")

    return [{"method": "npJIVE", "lambda": best_lambda, "gamma": best_gamma, "risk": cv_risk},
    {"method": "2SLS", "lambda": best_lambda_no_split, "gamma": best_gamma_no_split, "risk" : cv_risk_no_split}, use_npjiv]
 



 