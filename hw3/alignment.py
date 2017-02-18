
import numpy as np

from ._alignment import _calc_sw_matrix

# Constants

GAP_OPENING = -3  # Penalty for opening a gap
GAP_EXTENSION = -1  # Penalty for extending an already open gap

# Functions


def _calc_sw_traceback(seq1, seq2, sw_score, sw_path):
    # Calculate the traceback for Smith-Waterman

    # Find the best score in the matrix
    idx_max = np.argmax(sw_score)
    i, j = np.unravel_index(idx_max, sw_score.shape)

    # Now traceback
    align1 = []
    align2 = []
    while True:
        score = sw_score[i, j]
        if score <= 0 or i == 0 or j == 0:
            break
        this_dir = sw_path[i, j]

        # Diagonal
        if this_dir == 0:
            align1.append(seq1[j-1])
            align2.append(seq2[i-1])
            i -= 1
            j -= 1
        elif this_dir == 1:
            # Up
            align1.append('-')
            align2.append(seq2[i-1])
            i -= 1
        elif this_dir == 2:
            # Left
            align1.append(seq1[j-1])
            align2.append('-')
            j -= 1

    # We acumulate from right to left, so reverse to get the final alignment
    align1 = ''.join(reversed(align1))
    align2 = ''.join(reversed(align2))
    return align1, align2


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

    # Calculate the score and path matricies
    sw_score, sw_path = _calc_sw_matrix(seq1, seq2, score,
                                        gap_opening=gap_opening,
                                        gap_extension=gap_extension)
    if calc_traceback:
        # Use the score and path to traceback
        align1, align2 = _calc_sw_traceback(seq1, seq2, sw_score, sw_path)
    else:
        align1, align2 = None, None
    return np.max(sw_score), align1, align2
