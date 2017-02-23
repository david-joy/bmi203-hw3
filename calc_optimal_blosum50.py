#!/usr/bin/env python

""" Calculate the optimal BLOSUM50 penalties

Tries a grid combination of gap_opening and gap_extention penalties
and records the resulting scores
"""

# Imports
import time
import pathlib
import multiprocessing
from collections import namedtuple
from itertools import product

from hw3 import io
from hw3.consts import DATADIR, SCOREDIR, PROCESSES
from hw3.alignment import smith_waterman


# Constants
NEGPAIR_FILE = DATADIR / 'Negpairs.txt'
POSPAIR_FILE = DATADIR / 'Pospairs.txt'
SCORE_FILE = DATADIR / 'BLOSUM50'

GAP_OPENING_MIN = 1
GAP_OPENING_MAX = 20

GAP_EXTENSION_MIN = 1
GAP_EXTENSION_MAX = 5

POSPAIR_SCORES = SCOREDIR / 'PospairScores.txt'
NEGPAIR_SCORES = SCOREDIR / 'NegpairScores.txt'


ScoreItem = namedtuple(
    'ScoreItem', 'seq1, seq2, score, gap_opening, gap_extension')

# Functions


def calc_single_score(item):
    """ Calculate a single score

    Everything packed in a tuple to be compatible with multiprocessing

    :param item:
        The ScoreItem object with the seqences, score matrix, and
        gap parameters for this test
    :returns:
        The alignment score and gap parameters
    """
    align_score, _, _ = smith_waterman(
        item.seq1, item.seq2, item.score,
        gap_opening=item.gap_opening,
        gap_extension=item.gap_extension,
        calc_traceback=False)
    return (align_score, item.gap_opening, item.gap_extension)


def calc_all_scores(pair_file, pair_scores):
    score = io.read_score_matrix(SCORE_FILE)

    pair_scores = pathlib.Path(pair_scores)

    have_scores = set()

    if pair_scores.is_file():
        print('Reloading scores: {}'.format(pair_scores))
        # Replay so we don't calculate stuff twice
        needs_header = False
        with pair_scores.open('rt') as fp:
            for line in fp:
                line = line.split('#', 1)[0].strip()
                if line == '':
                    continue
                p1, p2, _, opening, extension = line.split(',')
                opening = int(opening)
                extension = int(extension)

                have_scores.add((p1, p2, opening, extension))
        print('Loaded {} scores'.format(len(have_scores)))

    else:
        needs_header = True

    with pair_scores.open('at') as fp:
        if needs_header:
            fp.write('#fasta1,fasta2,score,opening,extension\n')

        for p1, p2 in io.read_pair_file(pair_file):
            _, seq1 = io.read_fasta(DATADIR / p1)
            _, seq2 = io.read_fasta(DATADIR / p2)

            gap_opening_penalties = range(GAP_OPENING_MIN,
                                          GAP_OPENING_MAX+1)
            gap_extension_penalties = range(GAP_EXTENSION_MIN,
                                            GAP_EXTENSION_MAX+1)

            # Try all combinations of scores and write them to a file
            items = product(gap_opening_penalties,
                            gap_extension_penalties)
            items = [ScoreItem(seq1=seq1, seq2=seq2, score=score,
                               gap_opening=-opening, gap_extension=-extention)
                     for opening, extention in items
                     if (p1, p2, -opening, -extention) not in have_scores]
            if len(items) == 0:
                print('Already finished {}, {}'.format(p1, p2))
                continue

            t0 = time.perf_counter()
            print('Processing {}, {}'.format(p1, p2))
            with multiprocessing.Pool(PROCESSES) as pool:
                for res in pool.imap_unordered(calc_single_score, items):
                    align_score, opening, extension = res
                    print(res)
                    # Probably should be using csv writer here...
                    rec = f'{p1},{p2},{align_score},{opening},{extension}\n'
                    fp.write(rec)
            print('Finished in {} secs'.format(time.perf_counter() - t0))
            fp.flush()


def main():
    calc_all_scores(POSPAIR_FILE, POSPAIR_SCORES)
    calc_all_scores(NEGPAIR_FILE, NEGPAIR_SCORES)


if __name__ == '__main__':
    main()
