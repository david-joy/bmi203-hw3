import pathlib

from hw3 import io

import pytest


# Constants
THISDIR = pathlib.Path(__file__).resolve().parent
DATADIR = THISDIR.parent / 'data'

# Score matricies and several scores from random cells
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

# Sequence files, sequence name, sequence len, first 20 aas
SEQ_FILES = [
    ('prot-0004.fa', 'd1flp__ 1.1.1.1.2', 142, 'SLEAAQKSNVTSSWAKASAA'),
    ('prot-0008.fa', 'd1ash__ 1.1.1.1.33', 147, 'ANKTRELCMKSLEHAKVDTS'),
    ('prot-0915.fa', 'd1adn__ 7.39.1.1.1', 92, 'MKKATCLTDDQRWQSVLARD'),
]

# Pair file name, number of entries
PAIR_FILES = [
    ('Negpairs.txt', 50),
    ('Pospairs.txt', 50),
]


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


@pytest.mark.parametrize('filename,name,length,head', SEQ_FILES)
def test_reads_fasta(filename, name, length, head):

    filepath = DATADIR / 'sequences' / filename
    assert filepath.is_file()

    # Read the score matrix into a pandas dataframe
    res_name, res_seq = io.read_fasta(filepath)

    # Make sure we get the name and the sequence
    assert res_name == name
    assert res_seq[:len(head)] == head
    assert len(res_seq) == length


@pytest.mark.parametrize('filename,num_pairs', PAIR_FILES)
def test_reads_pair_file(filename, num_pairs):

    filepath = DATADIR / filename
    assert filepath.is_file()

    pairs = io.read_pair_file(filepath)

    assert len(pairs) == num_pairs
    for p1, p2 in pairs:
        p1path = DATADIR / p1
        assert p1path.is_file()

        p2path = DATADIR / p2
        assert p2path.is_file()
