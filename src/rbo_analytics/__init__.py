"""This module provides a series of analytics for hypothesis testing that 2 recommenders are related."""

from typing import List, Any, Set, Tuple
import numpy as np
from numpy.typing import NDArray

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
        if item_a not in A_k_set:
            if item_a in B_k_set or item_a == item_b:
                C_k_intersection.add(item_a)
        
        # Check if item_b is now in the intersection (only if it wasn't item_a)
        if item_a != item_b and item_b not in B_k_set:
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

# The inner calculation logic for RBO is complex. Your original logic appears to be 
# an efficient way to recursively track the size of the intersection set $|A[:k] \cap B[:k]|$ 
# without recomputing the full intersection at every step.

