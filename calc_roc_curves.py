#!/usr/bin/env python

""" Calculate ROC curves for each matrix

Figure out how each matrix responds to the positive and negative examples
we have
"""

# Imports
import time
import multiprocessing
from collections import namedtuple

# Our own imports
from hw3 import io
from hw3.consts import (
    DATADIR, SCOREDIR, GAP_OPENING, GAP_EXTENSION, PROCESSES)
from hw3.alignment import smith_waterman


# Constants
NEGPAIR_FILE = DATADIR / 'Negpairs.txt'
POSPAIR_FILE = DATADIR / 'Pospairs.txt'
SCORE_FILES = [
    DATADIR / 'BLOSUM50',
    DATADIR / 'BLOSUM62',
    DATADIR / 'MATIO',
    DATADIR / 'PAM100',
    DATADIR / 'PAM250',
]

OVERWRITE = False  # If True, delete the old score files. Else, reuse them

ScoreItem = namedtuple(
    'ScoreItem', 'p1, p2, seq1, seq2, score')

# Functions


def calc_single_score(item):
    """ Calculate a single score

    Everything packed in a tuple to be compatible with multiprocessing

    :param item:
        The ScoreItem object with the seqences, and score matrix
    :returns:
        fasta1, fasta2, alignment score, normalized score
    """
    min_len = min([len(item.seq1), len(item.seq2)])
    align_score, _, _ = smith_waterman(
        item.seq1, item.seq2, item.score,
        gap_opening=GAP_OPENING,
        gap_extension=GAP_EXTENSION,
        calc_traceback=False)
    return (item.p1, item.p2, align_score, align_score/min_len)


def read_pair_scores(pair_file):
    """ Generator yielding scores from a paired score file """

    for p1, p2 in io.read_pair_file(pair_file):
        _, seq1 = io.read_fasta(DATADIR / p1)
        _, seq2 = io.read_fasta(DATADIR / p2)
        yield p1, p2, seq1, seq2


def calc_all_scores(pair_file, score_file):
    """ Calculate the entire score file """

    pair_scores = score_file.name + '-' + pair_file.name
    pair_scores = SCOREDIR / pair_scores

    have_scores = set()

    if OVERWRITE and pair_scores.is_file():
        pair_scores.unlink()

    if pair_scores.is_file():
        print('Reloading scores: {}'.format(pair_scores))
        # Replay so we don't calculate stuff twice
        needs_header = False
        with pair_scores.open('rt') as fp:
            for line in fp:
                line = line.split('#', 1)[0].strip()
                if line == '':
                    continue
                p1, p2, _ = line.split(',')

                have_scores.add((p1, p2))
        print('Loaded {} scores'.format(len(have_scores)))

    else:
        needs_header = True

    score = io.read_score_matrix(score_file)

    with pair_scores.open('at') as fp:
        if needs_header:
            fp.write(f'#gap_opening={GAP_OPENING}\n')
            fp.write(f'#gap_extension={GAP_EXTENSION}\n')
            fp.write('#fasta1,fasta2,score,norm_score\n')

        items = (ScoreItem(p1=p1, p2=p2, seq1=seq1, seq2=seq2, score=score)
                 for p1, p2, seq1, seq2 in read_pair_scores(pair_file)
                 if (p1, p2) not in have_scores)

        t0 = time.perf_counter()
        print('Processing {}, {}'.format(pair_file.name, score_file.name))
        with multiprocessing.Pool(PROCESSES) as pool:
            for res in pool.imap_unordered(calc_single_score, items):
                p1, p2, align_score, norm_score = res
                print(res)
                # Probably should be using csv writer here...
                rec = f'{p1},{p2},{align_score},{norm_score}\n'
                fp.write(rec)
        print('Finished in {} secs'.format(time.perf_counter() - t0))
        fp.flush()


def main():
    for score_file in SCORE_FILES:
        calc_all_scores(POSPAIR_FILE, score_file)
    for score_file in SCORE_FILES:
        calc_all_scores(NEGPAIR_FILE, score_file)


if __name__ == '__main__':
    main()
