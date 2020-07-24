"""
    References:
    Ludmila I. Kuncheva. A stability index for feature selection. 
    In Artificial Intelligence and Applications (AIAP’07), pages 390–395, 2007.

"""

def get_consistency_index(A_subset: list, B_subset: list, n: int):
    # The sizes of both subsets must be equal and are hold by variable k
    # And the variable r represents the cardinality of (A ∩ B)

    k = len(A_subset)
    if k != len(B_subset):
        raise Exception('The given A and B subsets have different cardinalities.')
    r = len(list(set(A_subset).intersection(B_subset)))

    consistency_index = ((r * n) - (k * k)) / (k * (n - k))

    return consistency_index


# Provide either the thresholded ranks with the n
# Or a the full ranks with a threshold value
def get_kuncheva_index(subsets: list, n=None, threshold=None):
    # n represents the original set size
    # threshold is an integer representing the number of elements considered
    # the subsets will be selected respecting the closed interval [0, threshold]
    # If a threshold is needed, n isn't needed, otherwise it is
    
    if n is None:
        if threshold is None:
            raise Exception('Provide a threshold value or the size of the original set.')
        else:
            th = threshold
            n = len(subsets[0])
    
    else:
        if threshold is not None:
            th = threshold
        else:
            th = len(subsets[0])

    pairwise_ci = 0
    for i in range(len(subsets) - 1):
        A_subset = subsets[i][:th]
        for j in range(i+1, len(subsets)):
            B_subset = subsets[j][:th]
            pairwise_ci += get_consistency_index(A_subset, B_subset, n)
        

    k = len(subsets)
    kuncheva_index = ((2) / (k * (k-1))) * pairwise_ci

    return kuncheva_index
