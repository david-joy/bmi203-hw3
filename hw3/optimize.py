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


def calc_distribution(alignments, weights=None):
    """ Calculate an empirical distribution of aligned bases

    :param alignments:
        A list of (seq1, seq2) pairs where seq1 and seq2 were aligned with
        the ``smith_waterman()`` function
    :returns:
        The log2 odds of those sequence counts occuring in the set of
        alignments
    """
    if weights is None:
        weights = np.ones((len(alignments), ))
    assert len(alignments) == len(weights)

    # Counter with a pseudo-count for every pairing
    # This prevents taking log(a number < 1)
    counts = np.ones((len(ALPHABET), len(ALPHABET)))
    counts = pd.DataFrame(counts, columns=ALPHABET, index=ALPHABET)

    # Accumulate counts for every paired base in the empirical alignments
    for (seq1, seq2), weight in zip(alignments, weights):

        # Ignore anything with a weight that doesn't matter
        if weight < 1e-2:
            continue

        assert len(seq1) == len(seq2)
        seq1 = seq1.replace('-', '*')  # BLOSUM uses * not -
        seq2 = seq2.replace('-', '*')
        for a1, a2 in zip(seq1, seq2):
            # TODO: This double counts the diagonals
            # ...not sure if that's a good thing or a bad thing
            # TODO: Should we normalize by sequence length?
            counts.loc[a1, a2] += weight
            counts.loc[a2, a1] += weight

    return np.log2(counts)


def calc_score_gradient(pos_scores, neg_scores,
                        pos_align, neg_align):
    """ Calculate a score matrix update

    Try to increase the margin between positive and negative alignments

    :param pos_scores:
        The np.ndarray of positive alignment scores
    :param neg_scores:
        The np.ndarray of negative alignment scores   
    :param pos_align:
        The list of paired alignments for positive scores
    :param neg_align:
        The list of paired alignments for negative scores
    :returns:
        A score matrix gradient to update the score matrix
    """
    pos_scores = np.array(pos_scores)
    neg_scores = np.array(neg_scores)

    # We actually don't care about anything other than the negative
    # scores that are larger than the smallest positive score and
    # vice versa.

    # Hence weight the updates to emphasize alignments close to the
    # boundary.
    min_pos_score = np.min(pos_scores)
    max_pos_score = np.max(pos_scores)

    max_neg_score = np.max(neg_scores)
    min_neg_score = np.min(neg_scores)

    # Work out which scores are inliers and which are outliers
    score_std = np.std(np.concatenate([pos_scores, neg_scores]))

    # Make a triangle function that weights positive scores high that are near
    # or less than the descision boundary
    pos_mask = pos_scores >= max_neg_score
    pos_weights = np.ones_like(pos_scores, dtype=np.float)
    if max_pos_score > max_neg_score:
        pos_good_scores = (pos_scores[pos_mask] - max_neg_score) / score_std
        pos_good_scores[pos_good_scores > 1.0] = 1.0
        pos_weights[pos_mask] = 1 - pos_good_scores

    # Inverse triangle function, low scores that are near or greater than the
    # descision boundary weigh more
    neg_mask = neg_scores <= min_pos_score
    neg_weights = np.ones_like(neg_scores, dtype=np.float)
    if min_neg_score < min_pos_score:
        neg_bad_scores = (min_pos_score - neg_scores[neg_mask]) / score_std
        neg_bad_scores[neg_bad_scores > 1.0] = 1.0
        neg_weights[neg_mask] = 1 - neg_bad_scores

    # Now reweight the matrix with the positive and negative alignments
    pos_score = calc_distribution(pos_align, weights=pos_weights)
    neg_score = calc_distribution(neg_align, weights=neg_weights)

    # Update the score
    # Increase score for pairs in the positive alignments
    # Decrease score for pairs in the negative alignments
    return pos_score - neg_score
