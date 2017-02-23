import pathlib

import pytest

import numpy as np

import pandas as pd

from hw3 import io
from hw3.alignment import (
    smith_waterman, _calc_sw_matrix, _encode_sw_matrix)


# Constants
THISDIR = pathlib.Path(__file__).resolve().parent
DATADIR = THISDIR.parent / 'data'

# Sequence1, sequence2, score matrix file, align1, align2
SCORES = [
    ('AAA', 'AAA', 'BLOSUM50', 15, 'AAA', 'AAA'),
    ('AAARNCCCC', 'AANCCCC', 'BLOSUM62', 47, 'AARNCCCC', 'AA-NCCCC'),
]


# Tests


@pytest.mark.parametrize('seq1,seq2,scorefile,exp_score,align1,align2',
                         SCORES)
def test_smith_waterman(seq1, seq2, scorefile, exp_score, align1, align2):
    scorefile = DATADIR / scorefile
    assert scorefile.is_file()

    score = io.read_score_matrix(scorefile)

    res_score, res_align1, res_align2 = smith_waterman(
        seq1, seq2, score, gap_opening=-3, gap_extension=-1)

    assert res_score == exp_score
    assert res_align1 == align1
    assert res_align2 == align2


def test_calc_sw_matrix():

    # Test against Wikipedia's result
    # https://en.wikipedia.org/wiki/Smith%E2%80%93Waterman_algorithm#Example

    seq1 = 'TGTTACGG'
    seq2 = 'GGTTGACTA'

    score = [
        [3, -3, -3, -3],
        [-3, 3, -3, -3],
        [-3, -3, 3, -3],
        [-3, -3, -3, 3],
    ]
    score = pd.DataFrame(score, columns=['A', 'C', 'T', 'G'],
                         index=['A', 'C', 'T', 'G'])

    # Encode the string as an array
    enc_seq1, enc_seq2, enc_score = _encode_sw_matrix(seq1, seq2, score)

    exp_seq1 = np.array([2, 3, 2, 2, 0, 1, 3, 3])
    exp_seq2 = np.array([3, 3, 2, 2, 3, 0, 1, 2, 0])
    exp_score = np.array([
        [3, -3, -3, -3],
        [-3, 3, -3, -3],
        [-3, -3, 3, -3],
        [-3, -3, -3, 3],
    ])

    np.testing.assert_equal(enc_seq1, exp_seq1)
    np.testing.assert_equal(enc_seq2, exp_seq2)
    np.testing.assert_equal(enc_score, exp_score)

    # Calculate the scores
    sw_score, sw_path = _calc_sw_matrix(enc_seq1, enc_seq2, enc_score,
                                        gap_opening=-2,
                                        gap_extension=-2)

    # Reference scores
    exp_score = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 3, 1, 0, 0, 0, 3, 3],
        [0, 0, 3, 1, 0, 0, 0, 3, 6],
        [0, 3, 1, 6, 4, 2, 0, 1, 4],
        [0, 3, 1, 4, 9, 7, 5, 3, 2],
        [0, 1, 6, 4, 7, 6, 4, 8, 6],
        [0, 0, 4, 3, 5, 10, 8, 6, 5],
        [0, 0, 2, 1, 3, 8, 13, 11, 9],
        [0, 3, 1, 5, 4, 6, 11, 10, 8],
        [0, 1, 0, 3, 2, 7, 9, 8, 7],
    ])
    np.testing.assert_equal(sw_score, exp_score)

    res_score, align1, align2 = smith_waterman(
        seq1, seq2, score, gap_opening=-2, gap_extension=-2)

    assert res_score == 13
    assert align1 == 'GTT-AC'
    assert align2 == 'GTTGAC'


def test_calc_sw_matrix_affine():

    # Test the affine vs linear gap penalty

    # DNAfull scoring matrix
    # http://rosalind.info/glossary/dnafull/
    score = [
        [5, -4, -4, -4],
        [-4, 5, -4, -4],
        [-4, -4, 5, -4],
        [-4, -4, -4, 5],
    ]
    score = pd.DataFrame(score, columns=['A', 'T', 'G', 'C'],
                         index=['A', 'T', 'G', 'C'])

    seq1 = 'TACGGGCCCGCTAC'
    seq2 = 'TAGCCCTATCGGTCA'

    # Linear gap penalty
    # Nasty mini-gaps all over the place
    res_score, align1, align2 = smith_waterman(
        seq1, seq2, score, gap_opening=-1, gap_extension=-1)

    assert res_score == 39
    assert align1 == 'TACGGGCCCGCTA-C'
    assert align2 == 'TA---G-CC-CTATC'

    # Affine gap penalty
    # Grouped blocks of gaps, much nicer
    res_score, align1, align2 = smith_waterman(
        seq1, seq2, score, gap_opening=-5, gap_extension=-1)

    # Interestingly, this is (slightly) different from wikipedia
    # I always branch up before left, so I get a slightly different
    # (although still reasonable) local alignment
    assert res_score == 27
    assert align1 == 'TACGGGCCCGCTA'
    assert align2 == 'TA---GCCC--TA'
