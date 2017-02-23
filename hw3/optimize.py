""" Code to optimize the BLOSUM-like scoring matrix """

# Imports

import numpy as np


# Functions


def score_matrix_objective(positive_scores, negative_scores):
    """ Objective function for the optimizer

    :param positive_scores:
        The scores for positive examples
    :param negative_scores:
        The scores for negative examples
    :returns:
        The sum of true-positive rates for a set of false-positive levels in
        the range [0.0, 4.0]
    """

    cutoffs = np.percentile(negative_scores, [100, 90, 80, 70])
    positive_len = positive_scores.shape[0]

    score = 0.0
    for cutoff in cutoffs:
        score += np.sum(positive_scores >= cutoff) / positive_len
    return score
