import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple

def check_prefix_relationship(token_a: str, token_b: str) -> bool:
    """
    Checks if token_a and token_b satisfy the correspondence rule:
    1. token_a is the same string as token_b.
    2. token_a is a prefix of token_b.
    3. token_b is a prefix of token_a.

    Retuns -1 if no prefix relationship exists, 0 if an exact match, or 1 otherwise.
    """
    if token_a == token_b:
        return 0
    
    # Check if a is a prefix of b
    if len(token_a) < len(token_b) and token_b.startswith(token_a):
        return 1
    
    # Check if b is a prefix of a
    if len(token_b) < len(token_a) and token_a.startswith(token_b):
        return 1
        
    return -1

def find_maximum_one_to_one_matching(list_a: List[str], list_b: List[str]) -> List[Tuple[str, str]]:
    """
    Finds the maximum one-to-one matching between tokens in list_a and list_b
    prioritizing matches with the smallest length difference (closest match).

    This is solved by transforming the Maximum Weight Bipartite Matching problem
    into a Minimum Cost Assignment problem using the Hungarian algorithm.
    """
    
    n_a = len(list_a)
    n_b = len(list_b)
    
    if n_a == 0 or n_b == 0:
        return []

    # 1. Determine the size of the cost matrix
    # The assignment problem requires a square matrix. We use the larger dimension
    # and pad the smaller list conceptually.
    matrix_size = max(n_a, n_b)

    # 2. Define the maximum possible cost for non-matching pairs
    # The maximum length difference possible is matrix_size. We set the max cost
    # slightly higher than this to penalize non-matching pairs.
    MAX_COST = matrix_size**2 + 1 
    
    # Initialize the cost matrix with the maximum cost (representing no match)
    cost_matrix = np.full((matrix_size, matrix_size), MAX_COST, dtype=float)
    
    # 3. Populate the cost matrix
    # The cost C(x, y) is defined as the absolute length difference: |len(x) - len(y)|.
    # This minimizes the cost for closer matches (cost=0 for exact match).
    
    for i in range(n_a):
        token_a = list_a[i]
        for j in range(n_b):
            token_b = list_b[j]
            
            if check_prefix_relationship(token_a, token_b)>-1:
                # Cost is the absolute difference in length (minimum cost is preferred)
                cost = abs(len(token_a) - len(token_b))
                if cost == 0:
                    cost = -1
                cost_matrix[i, j] = cost
    
    # 4. Apply the Hungarian Algorithm (linear_sum_assignment)
    # The algorithm finds the assignment (row_ind, col_ind) that minimizes the total cost.
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # 5. Extract the resulting one-to-one matching pairs
    matching_pairs = []
    
    for r, c in zip(row_ind, col_ind):
        # r and c are the indices of the matched pair in the padded matrix.
        
        # Check if the matched indices correspond to real tokens
        if r < n_a and c < n_b:
            # Check if this match is *not* one of the penalized non-matches.
            # If the resulting cost is MAX_COST, it means the algorithm paired a 
            # real token with a 'dummy' token (or a very poor, non-prefix match) 
            # because it was forced to fill the square matrix. We only keep 
            # meaningful matches (cost < MAX_COST).
            if cost_matrix[r, c] < MAX_COST:
                matching_pairs.append((list_a[r], list_b[c]))
                
    return matching_pairs

# --- Example Usage ---
if __name__ == '__main__':
    # List A tokens
    tokens_a = [
        "app",      # Prefixes 'apple', Exact match 'pear'
        "apple",    # Exact match 'apple'
        "pear",     # Exact match 'pear'
        "banana",   # Matches 'bananas'
        "orange",   # No match
        "goo"       # Ambiguous: matches 'google' and 'google_search'
    ]
    
    # List B tokens
    tokens_b = [
        "apple",        # Exact match 'apple'
        "applesauce",   # Matches 'apple' (diff 5)
        "pear",         # Exact match 'pear'
        "bananas",      # Matches 'banana' (diff 1)
        "google",       # Matches 'goo' (diff 3)
        "google_search" # Matches 'goo' (diff 8)
    ]

    print(f"List A: {tokens_a}")
    print(f"List B: {tokens_b}")
    print("\n--- Running Maximum Weight Bipartite Matching ---\n")

    # The matching aims to resolve the ambiguities:
    # 1. 'app' vs 'apple'/'applesauce' vs 'pear' -> 'app' will prioritize 'apple' (diff 2) over 'applesauce' (diff 7)
    # 2. 'goo' vs 'google' (diff 3) vs 'google_search' (diff 8) -> 'goo' will prioritize 'google' (diff 3)
    # 3. 'apple' is an exact match for 'apple' (diff 0), making it the strongest possible pair.
    
    # Since the algorithm finds the *overall best* set of matches, 
    # 'apple' (A) will take 'apple' (B) due to the perfect cost (0). 
    # This leaves 'app' (A) to match with 'applesauce' (B) or 'pear' (B). 
    # The Hungarian algorithm will find the global optimum.
    
    matching = find_maximum_one_to_one_matching(tokens_a, tokens_b)

    print("Identified One-to-One Matching Pairs (A, B):")
    if matching:
        for a, b in matching:
            diff = abs(len(a) - len(b))
            print(f"  A: '{a}' ({len(a)}) <--> B: '{b}' ({len(b)}) [Length Diff: {diff}]")
    else:
        print("No matches found.")

    # Demonstration of the 'goo' ambiguity resolution:
    # 'goo' vs 'google' (diff 3)
    # 'goo' vs 'google_search' (diff 8)
    # The algorithm selects the pair with the *smallest length difference* globally.
    print("\n--- Example: Ambiguity Resolution ---\n")
    
    ambiguous_a = ["short", "shorter_token"]
    ambiguous_b = ["short", "shortest", "longer_token_match"] 
    # 'short' (A) can match 'short' (B) (diff 0) or 'shortest' (B) (diff 3) or 'longer_token_match' (B) (diff 13)
    # 'shorter_token' (A) can match 'short' (B) (diff 8) or 'shortest' (B) (diff 5) or 'longer_token_match' (B) (diff 5)
    
    # Optimal global matching will ensure 'short' (A) takes 'short' (B) (Cost 0), 
    # then 'shorter_token' (A) takes 'shortest' (B) or 'longer_token_match' (B) (Cost 5).
    
    matching_ambiguity = find_maximum_one_to_one_matching(ambiguous_a, ambiguous_b)
    
    print(f"List A: {ambiguous_a}")
    print(f"List B: {ambiguous_b}")
    print("\nIdentified One-to-One Matching Pairs (A, B):")
    for a, b in matching_ambiguity:
        diff = abs(len(a) - len(b))
        print(f"  A: '{a}' <--> B: '{b}' [Length Diff: {diff}]")
