#!/usr/bin/env python

""" Optimize the score matrix """

# Imports

import time

import numpy as np

from hw3 import io
from hw3 import optimize, alignment
from hw3.consts import DATADIR

# Constants

NEGPAIR_FILE = DATADIR / 'Negpairs.txt'
POSPAIR_FILE = DATADIR / 'Pospairs.txt'

SCORE_FILE = DATADIR / 'BLOSUM50'
GAP_OPENING = -7  # Penalty for opening a gap
GAP_EXTENSION = -3  # Penalty for extending an already open gap

ALPHA = 0.1  # Step size for the gradient
NUM_STEPS = 3  # Number of optimization loops

# Functions


def main():

    # Initialize the score matrix
    score = io.read_score_matrix(SCORE_FILE)

    # Optimization loops
    for n in range(NUM_STEPS + 1):

        print('Optimizing step {} of {}'.format(n, NUM_STEPS))

        print('Aligning Positive examples...')
        t0 = time.perf_counter()
        pos_results = []
        for i, item in enumerate(alignment.align_all(POSPAIR_FILE, score)):
            print('* {}'.format(i + 1))
            pos_results.append(item)
        dt = time.perf_counter() - t0
        print('Positive finished in {:0.1f} secs'.format(dt))

        print('Aligning Negative examples...')
        t0 = time.perf_counter()
        neg_results = []
        for i, item in enumerate(alignment.align_all(NEGPAIR_FILE, score)):
            print('* {}'.format(i + 1))
            neg_results.append(item)
        dt = time.perf_counter() - t0
        print('Negative finished in {:0.1f} secs'.format(dt))

        # Get a measure of how good our current step is
        pos_scores = [item.align_score for item in pos_results]
        neg_scores = [item.align_score for item in neg_results]

        opt = optimize.score_matrix_objective(pos_scores, neg_scores)
        print('Step {}: Opt Score {}'.format(n, opt))

        # Unpack the positive and negative scores
        pos_align = [(item.align1, item.align2) for item in pos_results]
        neg_align = [(item.align1, item.align2) for item in neg_results]

        # Calculate the empirical distributions
        print('Calculating update...')
        pos_score = optimize.calc_distribution(pos_align)
        neg_score = optimize.calc_distribution(neg_align)

        # Update the score
        # Increase score for pairs in the positive alignments
        # Decrease score for pairs in the negative alignments
        grad_score = pos_score - neg_score
        print('Gradient Mean: {}'.format(np.mean(np.abs(grad_score))))
        score += ALPHA * grad_score

        print(score)


if __name__ == '__main__':
    main()
