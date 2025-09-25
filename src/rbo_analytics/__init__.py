"""This module provides a series of analytics for hypothesis testing that 2 recommenders are related."""

from typing import List, Any, Set, Tuple, Union
import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm

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

    
