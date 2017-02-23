""" Code to optimize the BLOSUM-like scoring matrix """

# Imports

import numpy as np

import pandas as pd


# Constants

# The list of amino acids
ALPHABET = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L',
    'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'B', 'Z',
    'X',  '*']


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

    positive_scores = np.array(positive_scores)
    negative_scores = np.array(negative_scores)

    assert positive_scores.ndim == 1
    assert negative_scores.ndim == 1

    cutoffs = np.percentile(negative_scores, [100, 90, 80, 70])
    positive_len = positive_scores.shape[0]

    score = 0.0
    for cutoff in cutoffs:
        score += np.sum(positive_scores >= cutoff) / positive_len
    return score


def calc_distribution(alignments):
    """ Calculate an empirical distribution of aligned bases

    :param alignments:
        A list of (seq1, seq2) pairs where seq1 and seq2 were aligned with
        the ``smith_waterman()`` function
    :returns:
        The log2 odds of those sequence counts occuring in the set of
        alignments
    """

    # Counter with a pseudo-count for every pairing
    # This prevents taking log(a number < 1)
    counts = np.ones((len(ALPHABET), len(ALPHABET)))
    counts = pd.DataFrame(counts, columns=ALPHABET, index=ALPHABET)

    # Accumulate counts for every paired base in the empirical alignments
    for seq1, seq2 in alignments:
        assert len(seq1) == len(seq2)
        seq1 = seq1.replace('-', '*')  # BLOSUM uses * not -
        seq2 = seq2.replace('-', '*')
        for a1, a2 in zip(seq1, seq2):
            # TODO: This double counts the diagonals
            # ...not sure if that's a good thing or a bad thing
            # TODO: Should we normalize by sequence length?
            counts.loc[a1, a2] += 1
            counts.loc[a2, a1] += 1

    return np.log2(counts)
