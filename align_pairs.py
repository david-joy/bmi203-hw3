#!/usr/bin/env python

""" Align sequences using the best score matrix and gap penalties """

# Imports
import pathlib
import argparse

from hw3 import io
from hw3.consts import (
    DATADIR, ALIGNDIR, GAP_OPENING, GAP_EXTENSION, PROCESSES)
from hw3.alignment import align_all

# Constants
NEGPAIR_FILE = DATADIR / 'Negpairs.txt'
POSPAIR_FILE = DATADIR / 'Pospairs.txt'

SCORE_FILE = DATADIR / 'BLOSUM50'

POSALIGN_FILE = ALIGNDIR / 'pospairs_{}.fa'.format(SCORE_FILE.name)
NEGALIGN_FILE = ALIGNDIR / 'negpairs_{}.fa'.format(SCORE_FILE.name)

# Command line interface


def parse_args(args=None):
    """ Parse the command line arguments

    :param args:
        The argv list to parse or None for sys.argv
    :returns:
        The parsed arguments
    """
    parser = argparse.ArgumentParser('Align pairs of sequences')
    parser.add_argument('-s', '--score-file', type=pathlib.Path,
                        default=SCORE_FILE,
                        help='Path to the score matrix to use')
    parser.add_argument('--gap-opening', type=float, default=GAP_OPENING,
                        help='Gap opening penalty')
    parser.add_argument('--gap-extension', type=float, default=GAP_EXTENSION,
                        help='Gap extension penalty')
    parser.add_argument('--processes', type=int, default=PROCESSES,
                        help='Number of parallel processes to use')
    return parser.parse_args(args=args)


def main(args=None):
    args = parse_args(args=args)

    score = io.read_score_matrix(args.score_file)

    ALIGNDIR.mkdir(exist_ok=True, parents=True)

    # Force the gap penalties to be negative
    gap_opening = -abs(args.gap_opening)
    gap_extension = -abs(args.gap_extension)

    # Align the positive pairs
    with POSALIGN_FILE.open('wt') as fp:
        for res in align_all(POSPAIR_FILE, score,
                             gap_opening=gap_opening,
                             gap_extension=gap_extension,
                             processes=args.processes):
            p1, p2, name1, name2, align_score, align1, align2 = res
            fp.write(f'>{name1},{name2},{align_score}\n')
            fp.write(f'{align1}\n')
            fp.write(f'{align2}\n\n')

    # Align the negative pairs
    with NEGALIGN_FILE.open('wt') as fp:
        for res in align_all(NEGPAIR_FILE, score,
                             gap_opening=gap_opening,
                             gap_extension=gap_extension,
                             processes=args.processes):
            p1, p2, name1, name2, align_score, align1, align2 = res
            fp.write(f'>{name1},{name2},{align_score}\n')
            fp.write(f'{align1}\n')
            fp.write(f'{align2}\n\n')


if __name__ == '__main__':
    main()
