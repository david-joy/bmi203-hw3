#!/usr/bin/env python

""" Make plots of the ROC curves we found in ``calc_roc_curves.py``

Also makes violin plots of the distributions to compare the original and
optimized score matrices.
"""

# Imports
import argparse
import pathlib

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from hw3 import io
from hw3.consts import SCOREDIR, PLOTDIR, ALIGNDIR

sns.set()

# Constants

TRUE_POSITIVE_THRESHOLD = 0.7  # Cutoff True Positive rate

# Functions


def load_score_file(scorefile):
    """ Load the score file

    :param scorefile:
        The score file to read
    :returns:
        A pandas DataFrame with the scores
    """
    return pd.read_csv(str(scorefile),
                       skiprows=3,
                       names=['fasta1', 'fasta2',
                              'score', 'norm_score'])


def find_score_pairs(score_file):
    """ Find all the score files

    :param score_file:
        The path to the score file to check
    :returns:
        A generator of (posfile, negfile) pairs
    """
    targets = {score_file.name, score_file.name + '_OPT'}

    # Group the score files by matrix
    pairs = {}
    for fpath in SCOREDIR.iterdir():
        if fpath.suffix != '.txt':
            continue
        if '-' in fpath.name:
            key = fpath.name.split('-', 1)[0]
            if key in targets:
                pairs.setdefault(key, []).append(fpath)

    for key, (p1, p2) in sorted(pairs.items()):
        # If we get the negative file first, swap them
        if p1.name.endswith('Negpairs.txt'):
            p2, p1 = p1, p2
        yield p1, p2


def find_alignment_pairs(score_file):
    """ Find all the score files

    :param score_file:
        The path to the score file to check
    :returns:
        A generator of (posfile, negfile) pairs
    """
    targets = {score_file.name, score_file.name + '_OPT'}

    # Group the score files by matrix
    pairs = {}
    for fpath in ALIGNDIR.iterdir():
        if fpath.suffix != '.fa':
            continue
        if '_' in fpath.stem:
            key = fpath.stem.split('_', 1)[1]
            if key in targets:
                pairs.setdefault(key, []).append(fpath)

    for key, (p1, p2) in sorted(pairs.items()):
        # If we get the negative file first, swap them
        if p1.name.startswith('negpairs_'):
            p2, p1 = p1, p2
        yield p1, p2


def calc_roc(positive_scores, negative_scores):
    """ Calculate the reciever operating characteristic

    :param positive_scores:
        The vector of positive scores
    :param negative_scores:
        The vector of negative scores
    :returns:
        the false positive rate, the true positive rate
    """

    # Find the split points for each cutoff
    cutoff_min = np.min([positive_scores, negative_scores])
    cutoff_max = np.max([positive_scores, negative_scores])

    cutoffs = np.linspace(cutoff_min, cutoff_max, 200)

    # Using those cutoffs, calculate the empirical rates
    num_positive = positive_scores.shape[0]
    num_negative = negative_scores.shape[0]

    tp_rate = [1.0]
    fp_rate = [1.0]
    for cutoff in cutoffs:
        tp_rate.append(np.sum(positive_scores >= cutoff) / num_positive)
        fp_rate.append(np.sum(negative_scores >= cutoff) / num_negative)
    tp_rate.append(0.0)
    fp_rate.append(0.0)
    return np.array(fp_rate), np.array(tp_rate)


def plot_roc_curves(score_file, score_key, title, outname):
    """ Plot the ROC curves for a given score """

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    print('{} Smith-Waterman:\n'.format(title))
    for posfile, negfile in find_score_pairs(score_file):
        label = posfile.name.split('-', 1)[0]
        pos_scores = load_score_file(posfile)
        neg_scores = load_score_file(negfile)

        # Plot ROC for this matrix
        roc_fp, roc_tp = calc_roc(pos_scores[score_key],
                                  neg_scores[score_key])
        ax.plot(roc_fp, roc_tp, label=label)

        # Find the spread at 70% True Positives for each matrix
        mask = roc_tp >= TRUE_POSITIVE_THRESHOLD
        tp_lvl = np.min(roc_tp[mask])
        fp_lvl = np.min(roc_fp[mask])
        print('- {}: TP {:1.0%} FP {:1.0%}'.format(label, tp_lvl, fp_lvl))

    # Plot the line for perfect confusion
    ax.plot([0, 1], [0, 1], '--')

    ax.set_title('{} ROC for Smith Waterman'.format(title))
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])

    ax.legend()

    fig.savefig(str(PLOTDIR / outname))
    plt.close()


def plot_violins(score_file, score_key, title, outname):
    """ Plot distributions for the scores """

    data = {
        'Score': [],
        'Matrix': [],
        'Condition': [],
    }

    for posfile, negfile in find_score_pairs(score_file):
        label = posfile.name.split('-', 1)[0]
        pos_scores = load_score_file(posfile)
        neg_scores = load_score_file(negfile)

        len_scores = len(pos_scores[score_key])

        data['Score'].extend(pos_scores[score_key])
        data['Matrix'].extend([label for _ in range(len_scores)])
        data['Condition'].extend(['Pos' for _ in range(len_scores)])

        data['Score'].extend(neg_scores[score_key])
        data['Matrix'].extend([label for _ in range(len_scores)])
        data['Condition'].extend(['Neg' for _ in range(len_scores)])

    df = pd.DataFrame(data)

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    sns.violinplot(data=df, ax=ax,
                   x='Matrix', y='Score', hue='Condition',
                   split=True)
    ax.set_title('{} Score Distributions'.format(title))
    fig.savefig(str(PLOTDIR / outname))
    plt.close()


def plot_alignment_violins(score_file, title, outname):
    """ Plot length distribution of alignments """

    data = {
        'Length': [],
        'Matrix': [],
        'Condition': [],
    }

    for posfile, negfile in find_alignment_pairs(score_file):
        label = posfile.stem.split('_', 1)[1]

        pos_alignments = io.read_alignment_fasta(posfile)
        neg_alignments = io.read_alignment_fasta(negfile)

        len_alignments = len(pos_alignments)
        assert len(neg_alignments) == len_alignments

        data['Length'].extend([len(align[0]) for align in pos_alignments])
        data['Matrix'].extend([label for _ in range(len_alignments)])
        data['Condition'].extend(['Pos' for _ in range(len_alignments)])

        data['Length'].extend([len(align[0]) for align in neg_alignments])
        data['Matrix'].extend([label for _ in range(len_alignments)])
        data['Condition'].extend(['Neg' for _ in range(len_alignments)])

    df = pd.DataFrame(data)

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    sns.violinplot(data=df, ax=ax,
                   x='Matrix', y='Length', hue='Condition',
                   split=True)
    ax.set_title('{} Length Distributions'.format(title))
    fig.savefig(str(PLOTDIR / outname))
    plt.close()


def parse_args(args=None):
    parser = argparse.ArgumentParser('Plot ROC for optimized matrices')
    parser.add_argument('score_file',
                        help='Path to the unoptimized score matrix',
                        type=pathlib.Path)

    return parser.parse_args(args=args)


def main(args=None):
    args = parse_args(args=args)

    score_name = args.score_file.name

    plot_violins(score_file=args.score_file,
                 score_key='score',
                 title='{} Unnormalized'.format(score_name),
                 outname='violin_unnorm_{}.png'.format(score_name))

    plot_violins(score_file=args.score_file,
                 score_key='norm_score',
                 title='{} Normalized'.format(score_name),
                 outname='violin_norm_{}.png'.format(score_name))

    plot_alignment_violins(score_file=args.score_file,
                           title='{} Alignment'.format(score_name),
                           outname='violin_align_{}.png'.format(score_name))

    plot_roc_curves(score_file=args.score_file,
                    score_key='score',
                    title='{} Unnormalized'.format(score_name),
                    outname='roc_matrix_unnorm_{}.png'.format(score_name))

    print('')

    plot_roc_curves(score_file=args.score_file,
                    score_key='norm_score',
                    title='{} Normalized'.format(score_name),
                    outname='roc_matrix_norm_{}.png'.format(score_name))


if __name__ == '__main__':
    main()
