import numpy as np
from discreteNPIV.solve_primal_structural import *
from discreteNPIV.solve_dual_riesz import *
from discreteNPIV.npiv_2SLS import *
from discreteNPIV.utils import *

def npiv_dml(X, Z, Y, X_new, dtrain=None, cv_n_splits=10,
             num_lambda=20, num_gamma=10, lambda_grid=None, gamma_grid=None, adaptive = True, verbose=False):
    """
   Automatic Debiased Machine Learning (autoDML) estimator for Discrete Nonparametric Instrumental Variable (NPIV) estimation.

    This function implements an automatic debiasing procedure for estimating linear functionals of the structural function in discrete NPIV models. It involves estimating the solutions to primal inverse problem for the structural function and a dual inverse problem for a Riesz representer of the linear functional.  Nonparametric jackknife instrumental variable estimation (npJIVE) is used to remove bias that arises whe having many weak instruments. Uncorrected estimates based on two-stage least squares (2SLS) are also returned. 

    
    Parameters
    ----------
    X : ndarray of shape (n_samples, d)
        Covariate matrix for the main sample.
    Z : ndarray of shape (n_samples,)
        Discrete/categorical instrumental variable assignments. If K categories, Z should take values in [0, 1, ..., K-1] with at least two data points for each category.
    Y : ndarray of shape (n_samples,)
        Outcome variable.
    X_new : ndarray of shape (m_samples, d)
        New set of covariates for computing the mean of the structural function (defines the estimand).
    dtrain : dict, optional wi
        Dictionary containing the training dataset with keys {'X', 'Z', 'Y'}.
    cv_n_splits : int, default=10
        Number of splits for cross-validation.
    num_lambda : int, default=20
        Number of values in the lambda grid.
    num_gamma : int, default=10
        Number of values in the gamma grid.
    lambda_grid : ndarray, optional
        Grid of lambda values. If not provided, a default grid is generated.
    gamma_grid : ndarray, optional
        Grid of gamma values. If not provided, a default grid is generated.
    num_repeats : int, default=5
        Number of repetitions for cross-validation splits.
    verbose : bool, default=False
        Whether to print progress messages.

    Returns
    -------
    dict
        A dictionary containing results from npJIVE and 2SLS estimations, with keys:
            - 'estimate': Debiased estimation of the structural function.
            - 'estimate_plugin': Plugin estimation of the structural function.
            - 'estimate_IPW': Inverse Probability Weighting (IPW) estimation.
            - 'se': Standard error of the debiased estimator.
            - 'ci_lower': Lower bound of the 95% confidence interval.
            - 'ci_upper': Upper bound of the 95% confidence interval.
            - 'EIF': Estimated influence function for debiased estimation.
            - 'output_2sls': Dictionary containing similar outputs for the 2SLS estimator.
    """

    
    # Ensure Z is a numpy array
    X = np.asarray(X)
    Z = np.asarray(Z)
    Y = np.asarray(Y)
    X_new = np.asarray(X_new)
    if dtrain is None:
        dtrain = {'X': X, 'Z': Z, 'Y': Y}

    K = len(set(Z))  # Number of instrument groups
    Z = convert_Z(Z)
    dtrain['Z'] = convert_Z(dtrain['Z'])

    
    
    # Compute group proportions
    n_total = len(Z)
    group_counts = np.bincount(Z, minlength=K)
    W = group_counts / n_total

    # npJIVE Estimation for Structural Parameter
    theta_hat, fit_info_primal = npiv_structural(
        dtrain['X'], dtrain['Z'], dtrain['Y'], n_splits=cv_n_splits, num_lambda=num_lambda, 
        num_gamma=num_gamma, lambda_grid=lambda_grid, gamma_grid=gamma_grid, 
       adaptive = adaptive, verbose=verbose
    )
    theta_hat_jive = fit_info_primal['jive']
    theta_hat_2sls = fit_info_primal['2sls']
    h_hat = X @ theta_hat
    h_hat_new = X_new @ theta_hat

    # npJIVE Estimation for Riesz Representer
    b_hat, fit_info_dual = npiv_dual(
        dtrain['X'], dtrain['Z'], X_new, n_splits=cv_n_splits, num_lambda=num_lambda, 
        num_gamma=num_gamma, lambda_grid=lambda_grid, gamma_grid=gamma_grid, 
        adaptive = adaptive, verbose=verbose
    )
    b_hat_jive = fit_info_dual['jive']
    b_hat_2sls = fit_info_dual['2sls']
    beta_hat = X @ b_hat

    # Compute group means
    T_beta_hat = compute_means_jackknife(beta_hat, Z)
    

    # Compute EIF
    EIF = T_beta_hat * (Y - h_hat)

    # Standard Error
    se = np.sqrt(
        (np.std(EIF, ddof=1) ** 2 / len(EIF)) + 
        (np.std(h_hat_new, ddof=1) ** 2 / len(X_new))
    )

    # Estimates
    estimate_plugin = np.mean(X_new @ theta_hat_jive)
    estimates_IPW = np.mean(T_beta_hat * Y)
    estimate_debiased = np.mean(h_hat_new) + np.mean(EIF)
 

    ci_lower, ci_upper = estimate_debiased - 1.96 * se, estimate_debiased + 1.96 * se

 

    # npJIVE Output
    output_jive = {
        'estimate': estimate_debiased,
        'estimate_plugin': estimate_plugin,
        'estimate_IPW': estimates_IPW,
        'se': se,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'EIF': EIF,
    }

    # single-fold jive
      
    cross_folds = make_cross_folds(Z)
    T_beta_hat_0_jive = compute_group_means_fold(X @ b_hat_jive, Z, cross_folds, 0, K)



    EIF_single = (cross_folds == 1) * T_beta_hat_0_jive[Z] * (Y - h_hat) / np.mean((cross_folds == 1))
    estimate_debiased_single = np.mean(h_hat_new) +  np.mean(EIF_single)
    se_single = np.sqrt(
        (np.std(EIF_single, ddof=1) ** 2 / len(EIF_single)) + 
        (np.std(h_hat_new, ddof=1) ** 2 / len(X_new))
    )
    ci_lower_single, ci_upper_single = estimate_debiased_single - 1.96 * se_single, estimate_debiased_single + 1.96 * se_single
    output_jive_single = {
        'estimate': estimate_debiased_single,
        'se': se_single,
        'ci_lower': ci_lower_single,
        'ci_upper': ci_upper_single,
        'EIF': EIF_single,
    }

    # 2SLS Solutions   
    
    h_hat_2SLS = X @ theta_hat_2sls
    h_hat_2SLS_new = X_new @ theta_hat_2sls

    beta_hat_2sls = X @ b_hat_2sls
    q_hat_2sls = compute_group_means(beta_hat_2sls, Z, K)

    T_Y = compute_group_means(Y, Z, K) 
    T_h_hat = compute_group_means(h_hat_2SLS, Z, K)

    est_plugin_2sls = np.mean(h_hat_2SLS_new)
    est_ipw_2sls = np.sum(W * q_hat_2sls * T_Y)
    est_dml_2sls = est_plugin_2sls + np.sum(W * q_hat_2sls * (T_Y - T_h_hat))

    # EIF for 2SLS
    EIF_2sls = q_hat_2sls[Z] * (Y - h_hat_2SLS)
    se_2sls = np.sqrt((np.std(EIF_2sls, ddof=1) ** 2 / len(EIF)) + 
                      (np.std(h_hat_2SLS_new, ddof=1) ** 2 / len(X_new)))
    ci_lower_2sls, ci_upper_2sls = est_dml_2sls - 1.96 * se_2sls, est_dml_2sls + 1.96 * se_2sls

    output_2sls = {
        'estimate': est_dml_2sls,
        'estimate_plugin': est_plugin_2sls,
        'estimate_IPW': est_ipw_2sls,
        'se': se_2sls,
        'ci_lower': ci_lower_2sls,
        'ci_upper': ci_upper_2sls,
        'EIF': EIF_2sls
    }

    output_jive['output_2sls'] = output_2sls
    output_jive['output_jive_single'] = output_jive_single
    return output_jive




 

 