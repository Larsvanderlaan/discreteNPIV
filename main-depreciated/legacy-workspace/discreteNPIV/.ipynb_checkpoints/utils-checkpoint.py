import numpy as np
import scipy.linalg
from scipy.sparse.linalg import lsmr
import pandas as pd

def lstsq_fast(M, b, solver=None):
    '''
    Solves the least squares problem Mx = b using various solvers.

    Args:
        M (np.ndarray): The matrix (n x d) in the linear system.
        b (np.ndarray): The vector (n,) in the linear system.
        solver (str or None): The solver to use. Options are:
                      - 'iterative': Uses scipy.sparse.linalg.lsmr (Iterative Method)
                      - 'solve': Uses Normal Equations with np.linalg.solve
                      - 'np': Uses np.linalg.lstsq (SVD-based)
                      - 'scipy': Uses scipy.linalg.lstsq (SVD-based, LAPACK)
                      - None: Automatically selects the best solver.

    Returns:
        np.ndarray: The solution vector x.
    '''

     
    # Automatically select solver if not provided
    if solver is None:
        # iterative is much faster than np when num columns is larger than 70
        if M.shape[0] > 70:  
            solver = 'iterative'
        else:  # Smaller or square problems
            solver = 'np'

    if solver == 'iterative':
        # Use the iterative solver lsmr
        x = lsmr(M, b)[0]

    elif solver == 'solve':
        # Use Normal Equations (assuming M is full-rank)
        M_T_M = M.T @ M
        M_T_b = M.T @ b
        x = np.linalg.solve(M_T_M, M_T_b)

    elif solver == 'np':
        # Use NumPy's SVD-based solver
        x = np.linalg.lstsq(M, b, rcond=None)[0]

    elif solver == 'scipy':
        # Use SciPy's LAPACK-based solver
        x = scipy.linalg.lstsq(M, b, cond=None)[0]

    else:
        raise ValueError("Invalid solver. Choose from 'iterative', 'solve', 'np', 'scipy'.")

    return x
    
 


 

def compute_means_jackknife(X, Z, Y=None):
    """
    Compute leave-one-out (LOO) means for X and Y (if provided), stratified by Z.
    
    Parameters:
    - X: np.ndarray or pd.DataFrame (n x p)
    - Z: np.ndarray or pd.Series (n,)
    - Y: np.ndarray or pd.Series (n,) - Optional

    Returns:
    - T_X: LOO means for X (n x p numpy array)
    - T_Y: LOO means for Y (n-length numpy array) if Y is provided, otherwise None
    """
    Z = pd.Series(Z, name='Z')
    
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    
    if Y is not None:
        X_combined = X.copy()  # Make a copy of X before modifying
        X_combined['Y'] = Y
        X_combined['Z'] = Z
        feature_columns = X.columns  # All columns from the original X
    else:
        X_combined = X.copy()
        X_combined['Z'] = Z
        feature_columns = X.columns  # All columns from the original X
        
    # Calculate group-wise sums and sizes for X (and Y if present)
    group_stats = X_combined.groupby('Z').agg({col: ['sum', 'size'] for col in X_combined.columns if col != 'Z'})
    group_stats.columns = [f'{col}_{stat}' for col, stat in group_stats.columns]

    # Pre-compute group sums and sizes
    group_sum_X = group_stats.filter(like='_sum').loc[X_combined['Z']].values
    group_size = group_stats.filter(like='_size').iloc[:, 0].loc[X_combined['Z']].values

    # Compute LOO means for X
    T_X = (group_sum_X[:, :len(feature_columns)] - X.values) / (group_size[:, None] - 1)
    single_obs_mask = group_size == 1
    T_X[single_obs_mask, :] = X.values[single_obs_mask, :]
    
    if Y is not None:
        group_sum_Y = group_stats['Y_sum'].loc[X_combined['Z']].values
        T_Y = (group_sum_Y - X_combined['Y'].values) / (group_size - 1)
        T_Y[single_obs_mask] = X_combined['Y'].values[single_obs_mask]
        
        # Convert to numpy arrays
        T_X = T_X.astype(float)
        T_Y = np.array(T_Y, dtype=float)
        
        return T_X, T_Y
    
    # If Y is not provided, return only T_X as a numpy array
    T_X = T_X.astype(float)
    return T_X







def compute_group_means_fold(data, Z, folds, fold_val, K):
    mask = (folds == fold_val)
    Z_sub = Z[mask]
    data_sub = data[mask]

    if data_sub.ndim == 1:
        df = pd.DataFrame({'Z': Z_sub, 'val': data_sub})
        means_series = df.groupby('Z')['val'].mean()
        # Reindex to get K rows
        means_array = np.zeros(K, dtype=data_sub.dtype)
        means_array[means_series.index] = means_series.values
        return means_array
    else:
        df_dict = {'Z': Z_sub}
        for j in range(data_sub.shape[1]):
            df_dict[f'col{j}'] = data_sub[:, j]
        df = pd.DataFrame(df_dict)
        means_df = df.groupby('Z').mean()
        # Reindex to get K rows
        means_array = np.zeros((K, data_sub.shape[1]), dtype=data_sub.dtype)
        for j in range(data_sub.shape[1]):
            colmean = f'col{j}'
            if colmean in means_df:
                means_array[means_df.index, j] = means_df[colmean].values
        return means_array

        
def compute_group_means(data, Z, K):
    return compute_group_means_fold(data, Z, np.ones_like(Z), 1, K)
 



def convert_Z(arr):
        """Converts any array of instrument values to integers 0, ..., K-1 if needed."""
        arr = np.asarray(arr)
        unique_vals = np.unique(arr)
        # Check if already in the form 0, ..., len(unique_vals)-1
        if not np.issubdtype(unique_vals.dtype, np.integer) or not np.array_equal(unique_vals, np.arange(len(unique_vals))):
            if verbose:
                print("Warning: Instrument variable Z is not encoded as integers 0, ..., K-1. Converting to that format.")
            mapping = {old_val: new_val for new_val, old_val in enumerate(unique_vals)}
            arr = np.array([mapping[val] for val in arr])
        return arr


from sklearn.model_selection import StratifiedKFold
 

def make_cross_folds(Z, n_splits=2, random_state=None):
    unique_groups, group_indices = np.unique(Z, return_inverse=True)
    
    if n_splits > len(unique_groups):
        raise ValueError("n_splits cannot be greater than the number of unique values in Z.")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    cross_folds = np.zeros(len(Z), dtype=int)
    
    for fold, (_, test_idx) in enumerate(skf.split(Z, group_indices)):
        cross_folds[test_idx] = fold
    
    return cross_folds


 
 
 

 