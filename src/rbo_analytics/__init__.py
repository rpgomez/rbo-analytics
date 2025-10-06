"""This module provides a series of analytics for hypothesis testing that 2 recommenders are related."""

from typing import List, Any, Set, Tuple, Union, Optional
import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm
import scipy.stats

import .mapping

# Note: The 'typing' module is for type hints. For functions using NumPy arrays,
# 'numpy.typing' is also often helpful, though standard Python types (like List, float)
# are sufficient for the arguments and return type in this case.

def compute_rbo_score(
    list_a: List[Any], 
    list_b: List[Any], 
    p: float
) -> float:
    """
    Computes the RBO (Rank-Biased Overlap) score up to a depth K.

    The formula for the partial sum RBO score is:
    $$ S(A,B) = \frac{1-p}{1-p^K} \sum_{k=1}^K p^{k-1} \cdot \frac{|A[:k] \cap B[:k]|}{k} $$
    
    Args:
        list_a: The first ranked list of items.
        list_b: The second ranked list of items.
        p: The persistence parameter (a float between 0 and 1).
        
    Returns:
        The RBO score (float).
        
    Raises:
        ValueError: If lists are not of the same length or if p is out of range.
    """
    
    K: int = len(list_a)
    if K != len(list_b):
        raise ValueError("Input lists must be of the same length.")
    if not (0.0 < p < 1.0):
         raise ValueError("The persistence parameter 'p' must be between 0 and 1.")

    # p_powers: p^(k-1) for k=1 to K
    p_powers: NDArray[np.float64] = p**np.arange(K) 
    
    # Track items seen in each list and their intersection
    A_k_set: Set[Any] = set()
    B_k_set: Set[Any] = set()
    C_k_intersection: Set[Any] = set()
    
    # Store normalized intersection size at each depth k
    normalized_intersections: List[float] = []

    # Iterate through the lists to compute the intersection size |A[:k] \cap B[:k]|
    for k in range(K):
        item_a: Any = list_a[k]
        item_b: Any = list_b[k]

        # Update running sets with the new item at rank k+1 (index k)
        
        # Check if item_a is now in the intersection
        if item_a in B_k_set or item_a == item_b:
            C_k_intersection.add(item_a)
        
        # Check if item_b is now in the intersection (only if it wasn't item_a)
        if item_a != item_b:
            if item_b in A_k_set: # item_b must have been added at a prior step
                C_k_intersection.add(item_b) 
            
        A_k_set.add(item_a)
        B_k_set.add(item_b)
        
        # Calculate the normalized intersection size for depth k+1
        # The length of the intersection is len(C_k_intersection)
        # The depth is k+1
        normalized_intersection_at_k: float = len(C_k_intersection) / (k + 1)
        normalized_intersections.append(normalized_intersection_at_k)

    # Convert to NumPy array for vectorized operation
    partial_scores_ratios: NDArray[np.float64] = np.array(normalized_intersections)
    
    # The sum term: \sum_{k=1}^K p^{k-1} \cdot \frac{|A[:k] \cap B[:k]|}{k}
    # It's better to sum the largest (least significant, k=K) terms first for numerical stability, 
    # but since numpy.dot handles this efficiently and the original code comment implies 
    # summing 'least significant first', we'll rely on the standard numpy order 
    # and the original intent for the calculation order.
    
    sum_term: np.float64 = np.dot(p_powers, partial_scores_ratios)

    # Final RBO score: (1-p)/(1-p^K) * sum_term
    score: np.float64 = ((1.0 - p) / (1.0 - p**K)) * sum_term
    
    return float(score)

def perform_montecarlo(probs,K=None, T=10000,verbose=False):
    """Performs T simulations of resampling from (N>=K) ranked items based on the probability distribution
    probs. For each resample an rbo score is generated against the ranked list 1...K

    probs is list of sorted in descending order probabilities

    N is the length of probs, the size of the ranked list.
    
    K is the number of entries that matter in the ranked list. K cannot be greater than the length of probs. If
    K is not provided, then we set K = N

    verbose is a boolean that determines whether or not a tqdm progressbar is displayed.
    returns the sorted list of montecarlo rbo scores.
    """

    N = len(probs)
    if K is None:
        K = N

    p = 0.01**(1/K)
    list_a_array: NDArray[np.int_] = np.arange(K)
    
    list_b_array: NDArray[np.int_] = np.random.choice(
            a=N, 
            size=K, 
            replace=False, 
            p=probs
        )

    iterable = range(T)
    if verbose:
        iterable = tqdm(iterable,desc='Performing Monte Carlo Simulations')

    scores = []
    for t in iterable:
        list_b_array: NDArray[np.int_] = np.random.choice(
            a=N, 
            size=K, 
            replace=False, 
            p=probs
        )
        scores.append(compute_rbo_score(list_a_array,list_b_array,p))

    scores = np.array(scores)
    scores.sort()

    return scores
        
def estimate_mu_variance_rbo_score(
    probs: NDArray[np.float64], 
    K: int, 
    p: float, 
    T: int = 10000, 
    verbose: bool = False
) -> Tuple[float, float]:
    """
    Estimates the mean (mu) and variance (var) of the RBO score using Monte Carlo simulation.

    This simulates a comparison between a fixed base ranking (A) and a simulated 
    ranking (B) where the items are drawn based on provided probabilities.
    
    Args:
        probs: 1D NumPy array of probabilities (summing to 1) for selecting 
               an item from the base set N. N is the size of the population.
        K: The prefix length (depth) of the ranked lists to consider (A[:K], B[:K]).
        p: The persistence parameter (float between 0 and 1) for the RBO calculation.
        T: The number of Monte Carlo simulations (trials) to run. Defaults to 10,000.
        verbose: If True, uses tqdm to display a progress bar for the simulations.

    Returns:
        A tuple containing:
        - mu (float): The estimated mean RBO score, E[RBO(A, B)].
        - var (float): The estimated variance of the RBO scores, Var[RBO(A, B)].
        
    Raises:
        ValueError: If the total number of items (N) is less than the cutoff rank (K), 
                    as a ranking of length K cannot be generated from N items.
    """

    # The array is initialized to store the RBO scores from T trials.
    scores: NDArray[np.float64] = np.zeros(T, dtype=np.float64)

    # N is the size of the total item population (domain)
    N: int = probs.shape[0] 
    
    # Base recommender list 'A' is fixed as the first K indices (items 0 to K-1)
    # This assumes that 'A' is a perfect ranking of the K most important items
    # and that 'probs' is aligned with this ideal ranking.
    list_a: List[int] = list(np.arange(K)) 

    if N < K:
        raise ValueError(f"Population size (N={N}) is smaller than cutoff rank (K={K}). Cannot sample ranking B.")

    # Determine the iterable for the loop (with or without progress bar)
    iterations: Union[range, tqdm]
    if verbose:
        iterations = tqdm(range(T), desc='Conducting RBO simulations')
    else:
        iterations = range(T)

    for t in iterations:
        # Generate the ranked list 'B' by sampling K items without replacement,
        # weighted by the provided probabilities.
        list_b_array: NDArray[np.int_] = np.random.choice(
            a=N, 
            size=K, 
            replace=False, 
            p=probs
        )
        
        # Convert the sampled array to a Python list for the compute_rbo_score function
        list_b: List[Any] = list(list_b_array)
        
        # Compute the RBO score for the pair (list_a, list_b)
        scores[t] = compute_rbo_score(list_a, list_b, p)

    # Compute the sample mean and variance of the simulation scores
    mu: float = scores.mean()
    var: float = scores.var()

    return mu, var


# Type alias for clarity
FTestResult = Tuple[float, float]

def f_test_anova_2_groups(
    group1: NDArray[np.float64], 
    group2: NDArray[np.float64], 
) -> FTestResult:
    """
    Computes the one-way ANOVA F-test statistic and the corresponding p-value 
    for comparing the means of two independent populations (groups).
    
    This function calculates the F-ratio as the ratio of explained variance 
    (Between-Group Mean Square) to unexplained variance (Within-Group Mean Square).

    Args:
        group1: A 1D NumPy array representing the first population.
        group2: A 1D NumPy array representing the second population.

    Returns:
        A tuple (f_ratio, p_value) containing:
        - f_ratio (float): The computed F-test statistic.
        - p_value (float): The upper-tail probability (p-value) corresponding 
                           to the F-ratio.

    Raises:
        ValueError: If either group has fewer than 1 element.
    """
    
    n1: int = group1.shape[0]
    n2: int = group2.shape[0]

    if n1 < 1 or n2 < 1:
        # F-test is invalid if a group is empty.
        # This case is handled in find_elbow_f_test's loop range, but included here for robustness.
        return 0.0, 1.0 

    K: int = 2 # Number of groups
    N: int = n1 + n2
    
    # 1. Compute means and overall grand mean
    mu1: float = float(group1.mean())
    mu2: float = float(group2.mean())
    mu_grand: float = (n1 * mu1 + n2 * mu2) / N # Overall grand mean

    # 2. Explained Variance (Between-Group Mean Square - MSB)
    # Sum of Squares Between (SSB)
    SSB: float = n1 * (mu1 - mu_grand)**2 + n2 * (mu2 - mu_grand)**2
    # Degrees of Freedom 1 (df1)
    df1: int = K - 1 
    # Mean Square Between
    expl_var: float = SSB / df1 

    # 3. Unexplained Variance (Within-Group Mean Square - MSW)
    # Sum of Squares Within (SSW)
    # Note: np.var() uses N as the denominator by default (ddof=0).
    # n * group.var(ddof=0) is equivalent to Sum of Squares from the mean.
    SSW: float = n1 * float(group1.var()) + n2 * float(group2.var()) 
    # Degrees of Freedom 2 (df2)
    df2: int = N - K
    # Mean Square Within
    unexpl_var: float = SSW / df2
    
    # Check for perfect fit (SSW=0, MSW=0)
    if unexpl_var == 0.0:
        f_ratio = np.inf
        p_value = 0.0
    else:
        # 4. F-ratio (MSB / MSW) and p-value
        f_ratio: float = expl_var / unexpl_var
        # sf (Survival Function) is 1 - CDF, giving the upper tail probability (p-value)
        p_value: float = float(scipy.stats.f.sf(f_ratio, df1, df2)) 
        
    return f_ratio, p_value

def find_elbow_f_test(
    sorted_values: Union[List[float], NDArray[np.float64]], 
    p_cutoff: float = 0.01
) -> Optional[int]:
    """
    Detects the most likely location of an elbow in a sorted sequence by finding 
    the split point that maximizes the ANOVA F-test statistic (maximum mean difference).
    
    A Bonferroni correction is applied to control the family-wise error rate 
    due to multiple comparisons.

    Args:
        sorted_values: A monotonic array or list of values.
        p_cutoff: The nominal p-value cutoff (alpha). Defaults to 0.01.

    Returns:
        The 0-based index `t` (where $2 \le t \le N-2$) of the split into 
        two populations: `sorted_values[:t]` and `sorted_values[t:]`. 
        Returns None if no split is statistically significant after correction.
    """
    
    y: NDArray[np.float64]
    if isinstance(sorted_values, list):
        y = np.array(sorted_values, dtype=np.float64)
    else:
        y = sorted_values

    N: int = y.shape[0]
    
    # We need at least 4 data points for a valid split (2 in group 1, 2 in group 2)
    # Splits run from t=2 ([:2], [2:]) to t=N-2 ([:N-2], [N-2:])
    min_split_t: int = 2
    max_split_t: int = N - 1
    
    if N < 4:
        return None

    # 1. Compute F-test for all candidate splits
    # The loop range(1, len(sorted_values)) in the original code is too broad, 
    # as splits at t=1 (1 element vs N-1) or t=N-1 (N-1 elements vs 1) don't have enough 
    # degrees of freedom for the ANOVA F-test (df2 = N - 2). A minimum of 2 elements 
    # per group (t from 2 to N-2) is required for a stable test.
    
    # Limiting the range to ensure at least 2 elements in each group.
    f_ratio_p_values_list: List[FTestResult] = []
    
    for t in range(min_split_t, max_split_t): 
        f_ratio_p_values_list.append(
            f_test_anova_2_groups(y[:t], y[t:])
        )
    
    if not f_ratio_p_values_list:
        return None

    f_ratio_p_values: NDArray[np.float64] = np.array(f_ratio_p_values_list)
    
    # 2. Separate F-ratios and p-values
    # The `try/except` block is replaced by direct access after list aggregation.
    f_ratios: NDArray[np.float64] = f_ratio_p_values[:, 0]
    p_values: NDArray[np.float64] = f_ratio_p_values[:, 1]
    
    # 3. Apply Bonferroni Correction
    K_total: int = p_values.shape[0] # Number of tests performed
    scaled_p_cutoff: float = p_cutoff / K_total

    # 4. Find the most significant split
    # The original logic used np.argmax(f_ratio), which is correct for finding the 
    # most "extreme" statistic, as a higher F-ratio corresponds to a smaller p-value.
    
    # Get the index (0-based) within the *f_ratios* array.
    max_f_index: int = np.argmax(f_ratios) 
    
    # Get the minimum p-value corresponding to the max F-ratio
    min_p_value: float = p_values[max_f_index]

    # 5. Check Significance
    if min_p_value >= scaled_p_cutoff:
        return None

    # The index t of the split is the index within the loop range.
    # The loop started at min_split_t (t=2), so we adjust the array index.
    split_t_location: int = min_split_t + max_f_index
    
    return split_t_location


def determine_K_p(
    probs: Union[List[float], NDArray[np.float64]], 
    N_max: int = 200
) -> Tuple[Optional[int], Optional[float]]:
    """
    Determines an optimal ranked list length K using the elbow detection F-test 
    on sorted probabilities, and then calculates the Rank-Biased Overlap (RBO) 
    persistence parameter 'p'.

    K is determined by finding a statistically significant split point in the 
    descending sorted probabilities, suggesting a natural separation between 
    important and less important items.

    The parameter 'p' is chosen to ensure that the weight of the last rank K is 
    at most 0.01, according to the condition:
    $$ p^K = 0.01 \quad \implies \quad p = (0.01)^{1/K} $$

    Args:
        probs: A list or 1D array of non-negative probabilities (or scores) 
               representing item importance.
        N_max: The maximum number of top probabilities (ranks) to consider 
               for the elbow test. Defaults to 200.

    Returns:
        A tuple (K, p) containing:
        - K (Optional[int]): The determined optimal ranked list size, or None 
                             if no significant elbow was found.
        - p (Optional[float]): The calculated persistence parameter, or None 
                               if K could not be determined.
    """
    
    # Ensure input is a NumPy array for efficient sorting and slicing
    if isinstance(probs, list):
        probs_array: NDArray[np.float64] = np.array(probs, dtype=np.float64)
    else:
        probs_array = probs

    # 1. Prepare data for the elbow test: Sort and truncate
    # Sort in descending order and limit the analysis to the top N_max probabilities.
    sorted_probs: NDArray[np.float64] = np.sort(probs_array)[::-1][:N_max]

    # 2. Determine K using the elbow test
    # The find_elbow_f_test function returns the index 't' where the split occurs,
    # meaning the elbow is at the start of the second group (t).
    K: Optional[int] = find_elbow_f_test(sorted_probs)

    # 3. Compute p based on K (only if K is found and valid)
    if K is None or K < 1:
        return None, None
    
    # Correct calculation: p = (0.01)^(1/K)
    # This ensures that the weight of the k-th rank, p^(k-1), 
    # is weighted less than 0.01 for k > K.
    try:
        p: float = 0.01**(1/K)
    except (ZeroDivisionError, ValueError):
        # Should not occur if K >= 1, but included for robustness
        return K, None

    return K, p


def compute_recommender_test_statistic(lists_a, lists_b, probs_a, verbose=False):
    """
    Implements the test statistic for related performance for recommenders A & B for me.

    lists_a is a list of ranked lists of recommendations by recommender A.
    lists_b is a corresponding list of ranked lists by recommender B.
    probs_a is a list of lists of probabilities, one list of probabilities per each list of recommendations by recommender A
    .

    Each list in lists_a should have been ranked sorted by the corresponding probability list
    The test statistic is given as:

    $$X = \sum_n RBOScore(list_a[n], lists_b[n], probs_a[n])$$

    Under the  null hypothesis, if A and B are statistically related in rank predictions then 

    $$ X ~ Norm(\mu,\sigma^2) $$

    where 

    $$\mu = \sum_n \mu_n, \qquad \sigma^2 = \sum_n \sigma^2_n $$

    and $\mu_n, \sigma^2_n$ are estimated using function **estimate_mu_variance_rbo_score**.

    This function returns the normalized sigmage score $Z_0 = (X - \mu)/\sigma$

    Note that this test statistic is NOT symmetric in the order of recommenders. In this regard
    it is similar in behavior to the Kullback-Leibler divergence function.


    if recommender A and B are related if 

    $$ Pr(Z \ge Z_0) \ge 0.01, \qquad  Z ~ Norm(0,1), \quad \text{equivalently } \quad Z_0 \gtrapprox -2.33 $$
    """

    # For each each list of probabilities determine appropriate K and p to use.
    if verbose:
        iterable = tqdm(probs_a,desc='estimating Ks and ps')
    else:
        iterable = probs_a
    
    Ks_ps = [determine_K_p(probs) for probs in iterable]

    # Determine a mask for which lists are valid to  use:
    mask = [K_p[1] is not None  for K_p in Ks_ps]

    # now filter data based on mask.
    new_lists_a = [list_a for t,list_a in enumerate(lists_a) if mask[t]]
    new_lists_b = [list_b for t,list_b in enumerate(lists_b) if mask[t]]
    new_probs_a = [probs_a for t,probs_a in enumerate(probs_a) if mask[t]]
    Ks_ps = [K_p for K_p in Ks_ps if K_p[1] is not None]
    
    # Now determine \mu_n, \sigma^2_n for each list_a in new_lists_a.
    sorted_probs = [np.sort(probs)[::-1] for probs in new_probs_a]

    iterable = sorted_probs
    if verbose:
        iterable = tqdm(iterable,desc='Estimating mu_n, variance_n')
        
    mu_sigmas = [estimate_mu_variance_rbo_score(probs[t],Ks_ps[t][0],Ks_ps[t][1]) \
                 for t,probs in enumerate(new_probs) ]

    mu_sigmas = np.array(mu_sigmas)
    mu, var = mu_sigmas.sum(axis=0)
    sigma = var**0.5
    
    # Now score list_a and against list_b for each pairing.
    scores = []

    for t, list_a in enumerate(new_lists_a):
        K,p = Ks_ps[t]
        list_b = new_lists_b[t]
        probs = new_probs_a[t]
        scores.append(compute_rbo_score(list_a[:K],list_b[:K],p))


    statistic = sum(scores)
    Z = (statistic - mu)/sigma
    return Z

