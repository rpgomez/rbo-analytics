import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple
import collections
from typing import List, Optional, Tuple, Dict, Set
import string

def detect_begin_stream_token(lists_of_tokens: List[List[str]]) -> Optional[str]:
    """
    Analyzes sequences of tokens to determine if the underlying tokenizer 
    vocabulary includes a dedicated 'begin-new-stream' token (e.g., [CLS], <s>).

    This token is typically added by a tokenizer as the first token of every 
    new input string or sequence.

    Args:
        lists_of_tokens: A list where each element is a sequence of tokens 
                         (List[str]) produced by tokenizing a distinct input string.

    Returns:
        The unique begin-new-stream token (str) if one is consistently found 
        at the start of all non-empty sequences, otherwise returns None.
    """

    if not lists_of_tokens:
        return None
    
    # 1. Extract the first token from every non-empty sequence
    # Handles cases where an input list (token_list) might be empty.
    first_tokens: List[str] = [token_list[0] for token_list in lists_of_tokens if token_list]

    if not first_tokens:
        # If all input lists were empty, we cannot determine a consistent start token.
        return None
        
    # 2. Check if all collected first tokens are identical
    # By converting the list of first tokens to a set, we can check its size.
    # If the size is 1, all tokens are the same.
    unique_first_tokens: Set[str] = set(first_tokens)
    
    if len(unique_first_tokens) == 1:
        # Return the single unique token found. 
        # Using .pop() is efficient for a set known to have only one element.
        return unique_first_tokens.pop()
    else:
        # If the set size is > 1, the first token is not consistent across streams.
        return None

def detect_word_start_prefix(vocabulary: List[str]) -> Optional[Tuple[str, int]]:
    """
    Analyzes a tokenizer's vocabulary to detect the special non-alphanumeric 
    symbol used as a prefix to indicate the start of a new word (a common feature 
    in BPE or SentencePiece tokenizers, like ' ' or 'Ġ').

    The detection is based on finding the most frequent non-alphanumeric character 
    that appears as the first character of tokens, provided the token has content 
    after the prefix.

    Args:
        vocabulary: A list of string tokens comprising the tokenizer's vocabulary.

    Returns:
        A tuple (prefix_symbol, count) containing the detected symbol and 
        its frequency as a prefix, or None if no significant prefix is detected.
    """
    if not vocabulary:
        return None

    valid_alphanumeric=string.ascii_letters + string.digits
    
    # Use collections.Counter to track potential prefix symbols
    prefix_candidates: collections.Counter[str] = collections.Counter()
    
    for token in vocabulary:
        # We only look at tokens that are long enough to have both a prefix and content
        if not token or len(token) <= 1:
            continue
        
        first_char = token[0]
        
        # Check if the first character is NOT a standard alphanumeric character (A-Z, a-z, 0-9)
        # or a character like 'Ġ':
        if not first_char.isalnum() or first_char in ['Ġ']:
            # This character is a non-alphanumeric prefix candidate.
            prefix_candidates[first_char] += 1

    if not prefix_candidates:
        return None

    # Get the most common prefix candidate and its count
    most_common_candidate, count = prefix_candidates.most_common(1)[0]

    # Heuristic check: To be a dedicated word-start marker, it should prefix 
    # a significant portion of the vocabulary (e.g., more than 5%).
    if count / len(vocabulary) > 0.05:
        return most_common_candidate, count
    else:
        # If the most common non-alphanumeric prefix is too rare, 
        # it is likely just punctuation or a control token, not a dedicated word-start marker.
        return None

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
