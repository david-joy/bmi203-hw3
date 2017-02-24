#!/usr/bin/env python

""" Optimize the score matrix """

# Imports
import time
import pathlib
import argparse

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from hw3 import io
from hw3 import optimize, alignment
from hw3.consts import DATADIR, PLOTDIR

sns.set()

# Constants

NEGPAIR_FILE = DATADIR / 'Negpairs.txt'
POSPAIR_FILE = DATADIR / 'Pospairs.txt'

SCORE_FILE = DATADIR / 'BLOSUM50'

GAP_OPENING = -7  # Penalty for opening a gap
GAP_EXTENSION = -3  # Penalty for extending an already open gap

ALPHA = 0.1  # Step size for the gradient
ALPHA_DECAY = 0.75  # Amount to multiply alpha by when the optimizer fails
MIN_ALPHA = 1e-4  # Minimum step size before stopping
NUM_STEPS = 100  # Number of optimization loops

OPT_MAX = 4.0  # Best score, stop early if we get here
OPT_WINDOW = 3  # Number of optimization steps to average over
OPT_BOREDOM = 10  # Number of steps with no progress

# Functions


def parse_args(args=None):
    parser = argparse.ArgumentParser('Optimize a score matrix')
    parser.add_argument('-s', '--score-file', type=pathlib.Path,
                        default=SCORE_FILE,
                        help='Path to the score matrix to use')
    return parser.parse_args(args=args)


def main(args=None):

    args = parse_args(args=args)
    score_out_file = DATADIR / '{}_OPT'.format(args.score_file.name)
    score_plot_file = PLOTDIR / 'opt_trajectory_{}.png'.format(args.score_file.name)

    # Initialize the score matrix
    score = io.read_score_matrix(args.score_file)
    opt_history = []
    score_history = []

    alpha = ALPHA

    # Optimization loops
    for n in range(NUM_STEPS + 1):

        print('==== Optimizing step {} of {} ===='.format(n, NUM_STEPS))
        print('Alpha = {:0.5f}'.format(alpha))

        print('Aligning Positive examples...')
        t0 = time.perf_counter()
        pos_results = []
        for i, item in enumerate(alignment.align_all(POSPAIR_FILE, score)):
            if i % 10 == 0:
                print('* {}'.format(i + 1))
            pos_results.append(item)
        dt = time.perf_counter() - t0
        print('Positive finished in {:0.1f} secs\n'.format(dt))

        print('Aligning Negative examples...')
        t0 = time.perf_counter()
        neg_results = []
        for i, item in enumerate(alignment.align_all(NEGPAIR_FILE, score)):
            if i % 10 == 0:
                print('* {}'.format(i + 1))
            neg_results.append(item)
        dt = time.perf_counter() - t0
        print('Negative finished in {:0.1f} secs\n'.format(dt))

        # Get a measure of how good our current step is
        pos_scores = np.array([item.align_score for item in pos_results])
        neg_scores = np.array([item.align_score for item in neg_results])

        # Debug scores
        print('Best Positive Score:  {:0.2f}'.format(np.max(pos_scores)))
        print('Worst Positive Score: {:0.2f}'.format(np.min(pos_scores)))
        print('Confused Positives:   {:d}'.format(np.sum(pos_scores <= np.max(neg_scores))))
        print('')

        print('Best Negative Score:  {:0.2f}'.format(np.max(neg_scores)))
        print('Worst Negative Score: {:0.2f}'.format(np.min(neg_scores)))
        print('Confused Negatives:   {:d}'.format(np.sum(neg_scores >= np.min(pos_scores))))
        print('')

        opt = optimize.score_matrix_objective(pos_scores, neg_scores)

        if len(opt_history) > 3 and opt < sum(opt_history[-OPT_WINDOW:])/OPT_WINDOW:
            print('Got worse opt, dropping alpha...')
            alpha = alpha * ALPHA_DECAY
            print('Alpha is now: {:0.5f}'.format(alpha))
            print('')

        opt_history.append(opt)
        score_history.append(score)

        if opt >= OPT_MAX:
            print('Got perfect score!')
            break

        if len(opt_history) >= OPT_BOREDOM:
            if all([abs(o - opt_history[-1]) < 1e-5 for o in opt_history[-OPT_BOREDOM:]]):
                print('Score hasn\'t changed in {} steps. Exiting!'.format(OPT_BOREDOM))
                break

        print('Step {}: Opt Score {:0.2f}'.format(n, opt))
        print('Average of last 3: {:0.2f}'.format(sum(opt_history[-OPT_WINDOW:])/OPT_WINDOW))
        print('Last 5 Opt: {}'.format(', '.join(
            '{:0.2f}'.format(o) for o in opt_history[-5:])))
        print('')

        # Unpack the positive and negative scores
        pos_align = [(item.align1, item.align2) for item in pos_results]
        neg_align = [(item.align1, item.align2) for item in neg_results]

        # Calculate the empirical distributions
        print('Calculating update...')

        grad_score = optimize.calc_score_gradient(pos_scores, neg_scores,
                                                  pos_align, neg_align)
        score += alpha * grad_score

        if alpha < MIN_ALPHA:
            print('Reached minimal step size...')
            break
        print('')
        print('Gradient max:  {:0.4f}'.format(np.max(grad_score.values)))
        print('Gradient min:  {:0.4f}'.format(np.min(grad_score.values)))
        print('Gradient mean: {:0.4f}'.format(np.mean(grad_score.values)))
        print('Gradient std:  {:0.4f}'.format(np.std(grad_score.values)))

        print('')

    opt_history = np.array(opt_history)
    opt_steps = np.arange(1, opt_history.shape[0] + 1)

    # Look back through history and find our best matrix
    print('==== Final Results ====')
    best_opt_idx = np.argmax(opt_history)
    print('Best Round: {}'.format(best_opt_idx))
    print('Best Opt:   {}'.format(opt_history[best_opt_idx]))
    print('Best Score: {}'.format(score_history[best_opt_idx]))
    print('')
    print('Last Opt:   {}'.format(opt_history[-1]))
    print('Last Score: {}'.format(score_history[-1]))

    print('Writing best scoring matrix')
    io.write_score_matrix(score_out_file, score_history[best_opt_idx])

    print('Plotting the optimization trajectory')

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    ax.plot(opt_steps, opt_history)
    ax.plot([1, opt_history.shape[0] + 1], [OPT_MAX, OPT_MAX], '--')

    ax.set_title('Optimization Progress')
    ax.set_xlabel('Number of steps')
    ax.set_ylabel('Objective Score')

    plt.tight_layout()

    fig.savefig(str(score_plot_file))

    plt.close()


if __name__ == '__main__':
    main()
