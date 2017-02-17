import pathlib

from hw3 import io

import pytest


# Constants
THISDIR = pathlib.Path(__file__).resolve().parent
DATADIR = THISDIR.parent / 'data'

SCORE_MATRICES = [
    ('BLOSUM50', [
        ('A', 'A', 5),
        ('A', 'C', -1),
    ]),
    ('BLOSUM62', [
        ('A', 'A', 4),
        ('R', 'N', 0),
    ]),
    ('MATIO', [
        ('A', 'A', 0),
        ('A', 'C', -2),
        ('L', 'A', -1),
    ]),
    ('PAM100', [
        ('A', 'A', 4),
        ('N', 'N', 5),
    ]),
    ('PAM250', [
        ('A', 'A', 2),
        ('N', 'N', 2),
    ]),
]
SCORE_COLUMNS = {
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L',
    'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'B', 'Z',
    'X',  '*',
}


# Tests


@pytest.mark.parametrize('filename,scores', SCORE_MATRICES)
def test_reads_score_matrix(filename, scores):

    filepath = DATADIR / filename
    assert filepath.is_file()

    # Read the score matrix into a pandas dataframe
    mat = io.read_score_matrix(filepath)

    assert mat.shape == (24, 24)
    assert set(mat.columns) == SCORE_COLUMNS
    assert set(mat.index) == SCORE_COLUMNS

    # Make sure we can pull out some scores
    for row, col, exp in scores:
        assert mat.loc[row, col] == exp
