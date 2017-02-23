""" Functions for local alignment tasks """

# Imports
import multiprocessing
from collections import namedtuple

import numpy as np

from . import io
from .consts import DATADIR, GAP_OPENING, GAP_EXTENSION, PROCESSES

from ._alignment import (
    _encode_sw_matrix, _calc_sw_matrix, _calc_sw_traceback)

# Tuples for passing multiprocessing data back and forth
AlignmentItem = namedtuple(
    'AlignmentItem', ['p1', 'p2', 'name1', 'name2', 'seq1', 'seq2',
                      'score', 'gap_opening', 'gap_extension'])
AlignmentResult = namedtuple(
    'AlignmentResult', ['p1', 'p2', 'name1', 'name2',
                        'align_score', 'align1', 'align2'])

# Functions


def smith_waterman(seq1, seq2, score,
                   gap_opening=GAP_OPENING,
                   gap_extension=GAP_EXTENSION,
                   calc_traceback=True):
    """ Smith-Waterman local sequence alignment

    :param seq1:
        A string containing the first sequence, as returned from read_fasta
    :param seq2:
        A string containing the second sequence, as returned from read_fasta
    :param score:
        A pandas DataFrame as returned from read_score_matrix
    :returns:
        The best score, and two strings: alignment1, alignment2
        These are the highest scoring **local** alignment of the two
        sequences. In a tie, one of the highest scoring alignments will be
        returned.
    """
    # Convert from strings and dataframes to arrays
    # Python strings are slow
    enc_seq1, enc_seq2, enc_score = _encode_sw_matrix(seq1, seq2, score)

    # Calculate the score and path matricies
    sw_score, sw_path = _calc_sw_matrix(enc_seq1, enc_seq2, enc_score,
                                        gap_opening=gap_opening,
                                        gap_extension=gap_extension)
    if calc_traceback:
        # Use the score and path to traceback
        align1, align2 = _calc_sw_traceback(seq1, seq2, sw_score, sw_path)
    else:
        align1, align2 = None, None
    return np.max(sw_score), align1, align2


def convert_pairs_to_items(pair_file, score,
                           gap_opening=GAP_OPENING,
                           gap_extension=GAP_EXTENSION):
    """ Convert the records in a pair file to ScoreItems

    :param pair_file:
        The file containing paired FASTA files to align
    :param score:
        The score matrix to use to align
    :returns:
        A generator yielding AlignmentItems for each record. Each item can be
        aligned by calling `calc_single_alignment`
    """
    for p1, p2 in io.read_pair_file(pair_file):
        name1, seq1 = io.read_fasta(DATADIR / p1)
        name2, seq2 = io.read_fasta(DATADIR / p2)

        yield AlignmentItem(p1, p2, name1, name2, seq1, seq2,
                            score, gap_opening, gap_extension)


def align_pair(item):
    """ Calculate a single alignment

    Everything packed in a tuple to be compatible with multiprocessing

    :param item:
        The AlignmentItem object with the seqences, and score matrix
    :returns:
        The AlignmentResult object with the score and local alignment
    """
    align_score, align1, align2 = smith_waterman(
        item.seq1, item.seq2, item.score,
        gap_opening=item.gap_opening,
        gap_extension=item.gap_extension)
    return AlignmentResult(
        item.p1, item.p2, item.name1, item.name2,
        align_score, align1, align2)


def align_all(pair_file, score,
              gap_opening=GAP_OPENING,
              gap_extension=GAP_EXTENSION,
              processes=PROCESSES):
    """ Calculate all the alignments for a pair file

    :param pair_file:
        The file containing paired FASTA files to align
    :param score:
        The score matrix to use to align
    :returns:
        A generator yielding AlignmentResult tuples
    """
    items = convert_pairs_to_items(pair_file, score,
                                   gap_opening=gap_opening,
                                   gap_extension=gap_extension)
    with multiprocessing.Pool(processes) as pool:
        for res in pool.imap_unordered(align_pair, items):
            yield res
