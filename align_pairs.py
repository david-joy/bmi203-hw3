#!/usr/bin/env python

""" Align sequences using the best score matrix and gap penalties """

# Imports
import pathlib
import multiprocessing
import argparse
from collections import namedtuple

from hw3 import io
from hw3.alignment import smith_waterman

# Constants

DATADIR = pathlib.Path('./data').resolve()
ALIGNDIR = pathlib.Path('./alignments').resolve()

NEGPAIR_FILE = DATADIR / 'Negpairs.txt'
POSPAIR_FILE = DATADIR / 'Pospairs.txt'

SCORE_FILE = DATADIR / 'BLOSUM50'
GAP_OPENING = -7  # Penalty for opening a gap
GAP_EXTENSION = -3  # Penalty for extending an already open gap

PROCESSES = 8  # Number of CPUs to run in parallel

POSALIGN_FILE = ALIGNDIR / 'pospairs_{}.fa'.format(SCORE_FILE.name)
NEGALIGN_FILE = ALIGNDIR / 'negpairs_{}.fa'.format(SCORE_FILE.name)

ScoreItem = namedtuple(
    'ScoreItem', 'p1, p2, score, gap_opening, gap_extension')


# Functions


def calc_single_alignment(item):
    """ Calculate a single alignment

    Everything packed in a tuple to be compatible with multiprocessing

    :param item:
        The ScoreItem object with the seqences, and score matrix
    :returns:
        fasta1, fasta2, name1, name2, alignment score, align1, align2
    """
    name1, seq1 = io.read_fasta(DATADIR / item.p1)
    name2, seq2 = io.read_fasta(DATADIR / item.p2)

    align_score, align1, align2 = smith_waterman(
        seq1, seq2, item.score,
        gap_opening=item.gap_opening,
        gap_extension=item.gap_extension)
    return (item.p1, item.p2, name1, name2, align_score, align1, align2)

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
    items = [ScoreItem(p1, p2, score, gap_opening, gap_extension)
             for p1, p2 in io.read_pair_file(POSPAIR_FILE)]

    with POSALIGN_FILE.open('wt') as fp:
        with multiprocessing.Pool(args.processes) as pool:
            for res in pool.imap_unordered(calc_single_alignment, items):
                p1, p2, name1, name2, align_score, align1, align2 = res
                fp.write(f'>{name1},{name2},{align_score}\n')
                fp.write(f'{align1}\n')
                fp.write(f'{align2}\n\n')

    # And align the negative pairs
    items = [ScoreItem(p1, p2, score, gap_opening, gap_extension)
             for p1, p2 in io.read_pair_file(NEGPAIR_FILE)]

    with NEGALIGN_FILE.open('wt') as fp:
        with multiprocessing.Pool(args.processes) as pool:
            for res in pool.imap_unordered(calc_single_alignment, items):
                p1, p2, name1, name2, align_score, align1, align2 = res
                fp.write(f'>{name1},{name2},{align_score}\n')
                fp.write(f'{align1}\n')
                fp.write(f'{align2}\n\n')


if __name__ == '__main__':
    main()
