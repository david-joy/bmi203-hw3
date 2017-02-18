import pathlib

import pandas as pd


# Functions


def read_score_matrix(filepath):
    """ Read in an amino acid score matrix

    The matrix is expected to contain a header column with whitespace
    separated amino acid names, e.g.::

        A  R  N  D  C ...

    Followed by a square matrix giving pairwise scores for each substitution::

        A  R  N  D  C ...
        0  1  1 -6 -2 ...
        1 -1  1 -4 -5 ...

    This matrix indicates that A->A scores 0, A->R scores 1, R->A scores 1,
    etc. This matrix must be square, but does not have to be symmetric.

    An unlimited number of lines with a '#' comment are ignored.

    :param filepath:
        A text file containing a score matrix
    :returns:
        A pandas dataframe with the scores
    """

    filepath = pathlib.Path(filepath)

    header = None
    lines = []
    with filepath.open('rt') as fp:
        for line in fp:
            # Ignore lines with # and empty lines
            line = line.split('#', 1)[0].strip()
            if line == '':
                continue
            if header is None:
                header = line.split()
            else:
                lines.append([float(l) for l in line.split()])

    # Make sure the file is sane
    if header is None:
        raise ValueError('{} does not have a header line'.format(filepath))

    if len(lines) != len(header):
        err = 'Expected square matrix: Got {} columns but {} rows of scores'
        err = err.format(len(header), len(lines))
        raise ValueError(err)

    for row in lines:
        if len(row) != len(header):
            err = 'Expected square matrix: got {} header columns but {} data'
            err = err.format(len(header), len(row))
            raise ValueError(err)

    # QA passed, turn the file into a dataframe
    return pd.DataFrame(lines, index=pd.Index(header), columns=header)


def read_fasta(filepath):
    """ Read the fasta file

    Assumes a FASTA file formatted::

        >some sequence name
        AAAAAAA...
        AAAAAAA...
        AAAA

    .. note:: If more than one sequnece is present, only the first is returned

    :param filepath:
        The FASTA formatted text file containing a single sequence
    :returns:
        The sequence name, the sequence
    """
    filepath = pathlib.Path(filepath)

    header = None
    seq = []
    with filepath.open('rt') as fp:
        for line in fp:
            line = line.strip()
            if line == '':
                continue
            if line.startswith('>'):
                if header is None:
                    header = line[1:]
                else:
                    # Found a second sequence
                    break
            else:
                seq.append(line)
    return header, ''.join(seq).upper()


def read_pair_file(filepath):
    """ Read in a pair file

    :param filepath:
        Path to the pair file
    :returns:
        A list of paired files
    """
    filepath = pathlib.Path(filepath)

    pairs = []
    with filepath.open('rt') as fp:
        for line in fp:
            line = line.strip()
            if line == '':
                continue
            pairs.append(line.split())
    return pairs
