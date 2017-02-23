#!/usr/bin/env python

""" Plot the BLOSUM50 penalty matrix results

These results, as calculated in ``calc_optimal_blosum50.py`` are used to
determine the optimal set of gap_opening and gap_extension penalties given
a fixed true positive rate.
"""

# Imports
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from hw3.consts import SCOREDIR, PLOTDIR

sns.set()


# Constants

TRUE_POSITIVE_RATE = 0.7  # fraction of positive pairs above a threshold
POSPAIR_SCORES = SCOREDIR / 'PospairScores.txt'
NEGPAIR_SCORES = SCOREDIR / 'NegpairScores.txt'

# Functions


def read_scorefile(scorefile):
    """ Read in a score file

    :param scorefile:
        The path to the CSV score file
    :returns:
        A pandas DataFrame with the scores
    """
    return pd.read_csv(str(scorefile),
                       skiprows=1,
                       names=['fasta1', 'fasta2', 'score',
                              'opening', 'extension'])


def truepos_percentile(scores, rate=TRUE_POSITIVE_RATE):
    """ Return the score percentile for a given true positive rate

    :param scores:
        A numpy array of scores
    :param rate:
        A float between 0 and 1 indicating how many scores should be
        **larger** than the cutoff.
    :returns:
        The cutoff level that will yield the true positive rate in rate
    """
    return np.percentile(scores, [(1 - rate)*100])[0]


def main():
    # Load the scores in
    pos_df = read_scorefile(POSPAIR_SCORES)
    neg_df = read_scorefile(NEGPAIR_SCORES)

    PLOTDIR.mkdir(exist_ok=True, parents=True)

    # Plot marginal scores
    # Gap opening vs score
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.plot(pos_df['opening'], pos_df['score'], 'ro',
            label='Positive Pairs')
    ax.plot(neg_df['opening'], neg_df['score'], 'bo',
            label='Negative Pairs')

    ax.legend()
    ax.set_xlabel('Opening penalty')
    ax.set_ylabel('SW BLOSUM50 Score')

    fig.savefig(str(PLOTDIR / 'gap_opening.png'))

    plt.close()

    # Gap extension vs score
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.plot(pos_df['extension'], pos_df['score'], 'ro',
            label='Positive Pairs')
    ax.plot(neg_df['extension'], neg_df['score'], 'bo',
            label='Negative Pairs')

    ax.legend()
    ax.set_xlabel('Extension penalty')
    ax.set_ylabel('SW BLOSUM50 Score')

    fig.savefig(str(PLOTDIR / 'gap_extension.png'))

    plt.close()

    # Make a heatmap of the true positive cutoff
    pos_table = pos_df.pivot_table(values='score',
                                   index=['opening'],
                                   columns=['extension'],
                                   aggfunc=truepos_percentile)

    # Heatmap of the positive penalty 70% threshold
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    sns.heatmap(pos_table, annot=True, fmt="0.1f", ax=ax)

    title = 'Threshold Giving a {:0.0%} True-Positive Rate'
    title = title.format(TRUE_POSITIVE_RATE)
    ax.set_title(title)
    ax.set_xlabel('Gap Extension Penalty')
    ax.set_ylabel('Gap Opening Penalty')

    fig.savefig(str(PLOTDIR / 'pos_threshold.png'))

    plt.close()

    # Work out the false positive rate from this cutoff
    neg_levels = {}
    for i, opening in enumerate(pos_table.index):
        for j, extension in enumerate(pos_table.columns):
            neg_scores = neg_df.where(np.logical_and(
                neg_df['opening'] == opening,
                neg_df['extension'] == extension))['score'].dropna()

            # Work out how many counts are at or above the threshold
            threshold = pos_table.iloc[i, j]
            neg_count = sum(neg_scores >= threshold)
            neg_level = neg_count / neg_scores.shape[0]

            neg_levels.setdefault('opening', []).append(opening)
            neg_levels.setdefault('extension', []).append(extension)
            neg_levels.setdefault('level', []).append(neg_level)

    neg_levels = pd.DataFrame(neg_levels)
    neg_table = neg_levels.pivot_table(values='level',
                                       index=['opening'],
                                       columns=['extension'],
                                       aggfunc=np.sum)

    # Heatmap of the false positive rate given the true positive threshold
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    sns.heatmap(neg_table, annot=True, fmt="0.1%", ax=ax)

    title = 'False Positive Rate For a {:0.0%} True Positive Threshold'
    title = title.format(TRUE_POSITIVE_RATE)
    ax.set_title(title)
    ax.set_xlabel('Gap Extension Penalty')
    ax.set_ylabel('Gap Opening Penalty')

    fig.savefig(str(PLOTDIR / 'neg_level.png'))

    plt.close()

    # Minimum False Negative Rate(s)
    # Check for multiple because we don't have that many data points
    best_level = np.min(neg_levels['level'])
    best_mask = neg_levels['level'] == best_level

    neg_level_min = neg_levels.where(best_mask).dropna()

    print('Lowest False Negative Rates:')
    print(neg_level_min)


if __name__ == '__main__':
    main()
