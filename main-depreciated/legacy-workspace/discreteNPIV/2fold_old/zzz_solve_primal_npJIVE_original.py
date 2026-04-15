from discreteNPIV.utils import * 
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
import time
from scipy.sparse.linalg import eigsh
from discreteNPIV.npiv_2SLS import * 


def npiv_primal(X, Z, Y, K, cross_folds=None, num_lambda=20, num_gamma=10, 
                lambda_grid=None, gamma_grid=None, n_splits=10, num_repeats=10, verbose=False):
    """
    Solves the Nonparametric Instrumental Variable (NPIV) problem using cross-validation 
    to select the best regularization parameters and either npJIVE or a standard 2SLS approach.

    Args:
        X (numpy.ndarray): Feature matrix of shape (n, d).
        Z (numpy.ndarray): Instrumental variable, categorical array of shape (n,).
        Y (numpy.ndarray): Response variable of shape (n,).
        K (int): Number of instrument categories.
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
            - h_fun (function): Function that takes X as input and returns predictions X @ theta_hat.
            - cv_output (tuple): Output from cross-validation, containing selected parameters and fold structure.
    """
    
    # Perform cross-validation to select the best regularization parameters
    cv_output = cross_validate_JIVE(X, Z, Y, K, 
                                    num_lambda=num_lambda, num_gamma=num_gamma, 
                                    cross_folds=cross_folds, lambda_grid=lambda_grid, 
                                    gamma_grid=gamma_grid, n_splits=n_splits, 
                                    lambda_min=1e-8, verbose=verbose)

    params_jive, params_2sls, use_npjiv, cross_folds = cv_output

    # Solve the estimation problem using either npJIVE or a standard 2SLS approach
    if use_npjiv:
        theta_hat, h_fun = solve_primal_npJIVE(X, Z, Y, K, cross_folds=cross_folds, 
                                               lambda_K=params_jive['lambda'],  
                                               gamma=params_jive['gamma'], 
                                               num_repeats=num_repeats)
    else:
        theta_hat, h_fun = solve_primal_structural_2SLS(X, Z, Y, K, 
                                                 lambda_K=params_2sls['lambda'],  
                                                 gamma=params_2sls['gamma'])

    return theta_hat, h_fun, cv_output


  

 

def solve_primal_npJIVE(X, Z, Y, K, cross_folds=None, lambda_K=1e-3,  gamma=0, num_repeats=1, project_PD = False):
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
            - h_fun (function): Function that takes X as input and returns predictions X @ theta_hat.
    """
    n_total, d = X.shape
    group_counts = np.bincount(Z, minlength=K)
    W = group_counts / n_total  # shape (K,)
    
    # Detect whether cross_folds is a list of folds or a single fold array
    if isinstance(cross_folds, list) and all(isinstance(f, np.ndarray) for f in cross_folds):
        num_repeats = len(cross_folds)  # Use the length of the list
    elif cross_folds is not None:
        num_repeats = 1  # Treat as a single cross-validation fold

    # Initialize accumulators for averaging
    M_reg_sum = np.zeros((d, d))
    b_sum = np.zeros(d)

    for r in range(num_repeats):
        # Select the appropriate cross_folds for this repeat
        if isinstance(cross_folds, list):
            current_folds = cross_folds[r]
        else:
            current_folds = make_cross_folds(Z) if cross_folds is None else cross_folds

        start_time = time.time()
        # Compute fold means for each group
        X_fold0 = compute_group_means(X, Z, current_folds, 0, K)
        X_fold1 = compute_group_means(X, Z, current_folds, 1, K)
        Y_fold0 = compute_group_means(Y, Z, current_folds, 0, K)
        Y_fold1 = compute_group_means(Y, Z, current_folds, 1, K)
        #print(f"Time for means: {time.time() - start_time:.6f} seconds")

         
        #Y_fold = compute_group_means(Y, Z, 0*current_folds, 0, K)

        # Compute the cross-products efficiently
        start_time = time.time()
        M = (X_fold0.T * W) @ X_fold1 + (X_fold1.T * W) @ X_fold0  # shape (d, d)
        b = (X_fold0.T * W) @ Y_fold1 + (X_fold1.T * W) @ Y_fold0  # shape (d,)
        M = M/2
        b = b/2
        #M = (X_fold.T * W) @ X_fold  
        #b = (X_fold.T * W) @ Y_fold  

        print("condioning before: ", np.linalg.cond(M), eigsh(M, k=1, which='SA', return_eigenvectors=False)[0])
        
        if project_PD:
            # ensure conditioning is as good as non-JIVE method
             
            X_fold = compute_group_means(X, Z, 0*current_folds, 0, K)
            M_full = (X_fol.T * W) @ X_fold 
            min_eigenvalue = eigsh(M_full, k=1, which='SA', return_eigenvectors=False)[0]
            M = project_to_pd(M, min_eigenvalue)
            print("condioning after: ", np.linalg.cond(M) , np.linalg.cond(M_full), min_eigenvalue)
        
        S = (X.T * W[Z]) @ X  # shape (d, d), weighted sum over all observations

        # Compute regularized M
        M_reg = M + lambda_K * S +  gamma * np.eye(d)


        
        # Accumulate values for averaging
        M_reg_sum += M_reg
        b_sum += b

    # Average M_reg and b across repeats
    M_reg_avg = M_reg_sum / num_repeats
    b_avg = b_sum / num_repeats

    # Solve the final system only once
    theta_hat = np.linalg.lstsq(M_reg_avg, b_avg, rcond=None)[0]
    #print(f"Time for solution: {time.time() - start_time:.6f} seconds")

    
    def h_fun(X):
        return X @ theta_hat

    return theta_hat, h_fun


def compute_JIVE_risk(preds, Z, X, Y, cross_folds, K, lambda_min=1e-8):
    """
    Computes the npJIVE risk (loss function) given an estimated parameter theta_hat.
    Averages risk across multiple cross-validation folds if cross_folds is a list.

    Args:
        preds (numpy.ndarray): Predicted values of shape (n,).
        Z (numpy.ndarray): Instrument variable, categorical array of shape (n,).
        X (numpy.ndarray): Feature matrix of shape (n, d).
        Y (numpy.ndarray): Response variable of shape (n,).
        cross_folds (list or numpy.ndarray): List of cross-validation folds or a single fold array.
        K (int): Number of instrument categories.
        lambda_min (float, optional): Regularization term to prevent division issues (default: 1e-5).

    Returns:
        float: Averaged computed risk value.
    """
    n_total = len(Z)
    group_counts = np.bincount(Z, minlength=K)
    W = group_counts / n_total  # Group weights

    # Detect whether cross_folds is a list of folds or a single fold array
    if isinstance(cross_folds, list) and all(isinstance(f, np.ndarray) for f in cross_folds):
        num_repeats = len(cross_folds)  # Use the length of the list
    elif cross_folds is not None:
        num_repeats = 1  # Treat as a single cross-validation fold
    else:
        raise ValueError("cross_folds must be a list of arrays or a single array.")

    risk_sum = 0.0  # Accumulate risks for averaging

    for r in range(num_repeats):
        # Select the appropriate cross_folds for this repeat
        if isinstance(cross_folds, list):
            current_folds = cross_folds[r]
        else:
            current_folds = cross_folds

        # Compute group means for errors
        err_group_1 = compute_group_means(Y - preds, Z, current_folds, 1, K)
        err_group_0 = compute_group_means(Y - preds, Z, current_folds, 0, K)

        # Compute JIVE risk
        jive_risk = np.mean(W * err_group_1 * err_group_0) / np.mean(W)
        jive_risk += lambda_min * np.mean(preds**2)

        risk_sum += jive_risk  # Accumulate risk values

    # Return the averaged risk across repeats
    return risk_sum / num_repeats


 
import numpy as np
from sklearn.model_selection import StratifiedKFold
from scipy.interpolate import interp1d
 

def cross_validate_JIVE(X, Z, Y, K, num_lambda=20, num_gamma=10, cross_folds=None,
                        lambda_grid=None, gamma_grid=None, n_splits=10, lambda_min=1e-8,   verbose=False):
    """
    Performs joint K-fold cross-validation to select the optimal regularization parameters (lambda_K, gamma) for npJIVE.
    - Iterates over lambda_K and gamma **in descending order**.
    - Stops decreasing lambda_K **on the first risk increase** (early stopping).
    - Stops searching gamma **once the best lambda's risk increases**.
    - Uses quadratic interpolation to refine gamma selection (similar to cvglmnet).
    
    Returns:
        tuple:
            - best_lambda (float): Selected lambda_K.
            - best_gamma (float): Selected gamma via interpolation.
    """

    
    # Initialize cross-validation folds if not provided
    num_repeats = 1
     

    
    # Define grids in **descending** order
    lambda_grid = np.logspace(-1, -8, num_lambda) if lambda_grid is None else np.sort(lambda_grid)[::-1]
    gamma_grid = np.logspace(-1, -6, num_gamma) if gamma_grid is None else np.sort(gamma_grid)[::-1]
    num_lambda = len(lambda_grid)
    num_gamma = len(gamma_grid)
    
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    n_total, d = X.shape

     

    group_counts = np.bincount(Z, minlength=K)
    W = group_counts / n_total  # shape (K,)
    
    start_time = time.time()
    # **Precompute Training/Validation Splits and Group Means**
    fold_data = []
    for train_idx, val_idx in kf.split(X, Z):
        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]
        Z_train, Z_val = Z[train_idx], Z[val_idx]
        if cross_folds is not None: 
            cross_folds_train = cross_folds[train_idx]
            cross_folds_val = cross_folds[val_idx]
        else:
            # if not provided, evaluate random splitting as part of CV
            cross_folds_train = make_cross_folds(Z_train)
            cross_folds_val = make_cross_folds(Z_val)
        #print(np.array([[np.mean(Z_val[cross_folds_val_direct==1] == k), np.mean(Z_val[cross_folds_val_direct==0] == k)] for k in range(10)]))

        
         
    
        # Compute and store group means
        X_avg_train = compute_group_means(X_train, Z_train, np.zeros_like(Z_train), 0, K)
        X_fold0_train = compute_group_means(X_train, Z_train, cross_folds_train, 0, K)
        X_fold1_train = compute_group_means(X_train, Z_train, cross_folds_train, 1, K)
        Y_avg_train = compute_group_means(Y_train, Z_train, np.zeros_like(Z_train), 0, K)
        Y_fold0_train = compute_group_means(Y_train, Z_train, cross_folds_train, 0, K)
        Y_fold1_train = compute_group_means(Y_train, Z_train, cross_folds_train, 1, K)

        # Precompute matrix components for efficiency
        M_base = (X_fold0_train.T * W) @ X_fold1_train + (X_fold1_train.T * W) @ X_fold0_train  # shape (d, d)
        b_base = (X_fold0_train.T * W) @ Y_fold1_train + (X_fold1_train.T * W) @ Y_fold0_train  # shape (d,)
        M_base = M_base/2
        b_base = b_base/2
        M_base_no_split = (X_avg_train.T * W) @ X_avg_train 
        b_base_no_split = (X_avg_train.T * W) @ Y_avg_train

        X_fold0_val = compute_group_means(X_val, Z_val, cross_folds_val, 0, K)
        X_fold1_val = compute_group_means(X_val, Z_val, cross_folds_val, 1, K)
        Y_fold0_val = compute_group_means(Y_val, Z_val, cross_folds_val, 0, K)
        Y_fold1_val = compute_group_means(Y_val, Z_val, cross_folds_val, 1, K)

        
        fold_data.append((train_idx, val_idx, X_val, M_base, b_base, X_fold0_val, X_fold1_val, Y_fold0_val, Y_fold1_val, M_base_no_split, b_base_no_split))
    if verbose:
        print(f"Time to compute fold means: {time.time() - start_time:.6f} seconds")
    # **Loop over gamma first, then lambda**
    start_time = time.time()

    

    # Storage for risk values
    risks = np.full((num_gamma, num_lambda), np.inf)
    best_lambda_risks = np.full(num_gamma, np.inf)
    risks_no_split = np.full((num_gamma, num_lambda), np.inf)
    best_lambda_risks_no_split = np.full(num_gamma, np.inf)
    
    # Precompute S (same for all folds)
    S = (X.T * W[Z]) @ X  # shape (d, d)
    for j, gamma in enumerate(gamma_grid):
        min_risk = np.inf
        last_valid_lambda = None
        last_valid_risk = np.inf
        break_split = False
        
        min_risk_no_split = np.inf
        last_valid_lambda_no_split = None
        last_valid_risk_no_split = np.inf
        break_no_split = False

        for i, lambda_K in enumerate(lambda_grid):
            preds = np.zeros(n_total)
            risk = 0
            risk_no_split = 0
            for (train_idx, val_idx, X_val, M_base, b_base, X_fold0_val, X_fold1_val, Y_fold0_val, Y_fold1_val, M_base_no_split, b_base_no_split) in fold_data:
                if not break_split:
                    M_reg = M_base + lambda_K * S + gamma * np.eye(d)  # Regularized matrix
                    theta_hat = np.linalg.lstsq(M_reg, b_base, rcond=None)[0]
                    risk += np.sum(W * (X_fold0_val @ theta_hat - Y_fold0_val) * (X_fold1_val @ theta_hat - Y_fold1_val)) 
                if not break_no_split:
                    M_reg_no_split = M_base_no_split + lambda_K * S + gamma * np.eye(d)  # Regularized matrix
                    theta_hat_no_split = np.linalg.lstsq(M_reg_no_split, b_base_no_split, rcond=None)[0]
                    risk_no_split += np.sum(W * (X_fold0_val @ theta_hat_no_split - Y_fold0_val) * (X_fold1_val @ theta_hat_no_split - Y_fold1_val))  
                
            risk /= n_splits
            risk_no_split /= n_splits

            #if verbose == 2:
            #    print(f"Gamma: {gamma:.2e}, Lambda: {lambda_K:.2e}, Risk: {risk:.6f},  break: {break_split:.6f}")


            if break_split or (risk > min_risk or risk < -1 / n_total):
                break_split = True  # Early stopping on first risk increase
            else:
                last_valid_lambda = lambda_K
                last_valid_risk = risk
                min_risk = risk
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
    if verbose:
        print(f"Time for select tunings: {time.time() - start_time:.6f} seconds")
    # Find best (gamma, lambda) pair
    
    min_gamma_idx = np.argmin(best_lambda_risks)
    best_gamma = gamma_grid[min_gamma_idx]
    best_lambda = lambda_grid[np.argmin(risks[min_gamma_idx, :])]
    cv_risk = np.min(risks[min_gamma_idx, :])

    min_gamma_idx_no_split = np.argmin(best_lambda_risks_no_split)
    best_gamma_no_split = gamma_grid[min_gamma_idx_no_split]
    best_lambda_no_split = lambda_grid[np.argmin(risks_no_split[min_gamma_idx_no_split, :])]
    cv_risk_no_split = np.min(risks_no_split[min_gamma_idx_no_split, :])
    
    use_npjiv = cv_risk <= cv_risk_no_split
     

    if cross_folds is None:
        cross_folds = make_cross_folds(Z)

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
    {"method": "2SLS", "lambda": best_lambda_no_split, "gamma": best_gamma_no_split, "risk" : cv_risk_no_split},
    use_npjiv,
    cross_folds]



