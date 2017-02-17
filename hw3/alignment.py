
import numpy as np


# Constants

GAP_OPENING = -3  # Penalty for opening a gap
GAP_EXTENTION = -1  # Penalty for extending an already open gap

# Functions


def _calc_sw_matrix(seq1, seq2, score,
                    gap_opening=GAP_OPENING,
                    gap_extension=GAP_EXTENTION):
    # Calculate the score matrix for Smith-Waterman

    # Initialize a score matrix with rows for seq2, cols for seq1
    rows = len(seq2)
    cols = len(seq1)
    sw_score = np.zeros((rows + 1, cols + 1))
    sw_path = np.zeros((rows + 1, cols + 1))

    # Scan through the matrix from top-left to bottom-right
    for i in range(1, rows+1):
        ci = seq2[i-1]
        for j in range(1, cols+1):
            cj = seq1[j-1]

            # Work out the gap penalty
            prev_dir_up = sw_path[i-1, j]
            prev_dir_left = sw_path[i, j-1]

            # Was the last up base a match?
            if prev_dir_up == 0:
                penalty_up = gap_opening
            else:
                # We're just extending...
                penalty_up = gap_extension

            # Was the last left base a match?
            if prev_dir_left == 0:
                penalty_left = gap_opening
            else:
                # Extending the gap again
                penalty_left = gap_extension

            # Work out the best of:
            # match the characters (diag)
            # gap in seq1 (top)
            # gap in seq2 (left)
            diag = sw_score[i-1, j-1] + score.loc[ci, cj]
            up = sw_score[i-1, j] + penalty_up
            left = sw_score[i, j-1] + penalty_left

            # The new direction is 0 - diagonal, 1 - up, 2 - left
            this_dir = np.argmax([diag, up, left])
            this_score = np.max([diag, up, left, 0])

            sw_score[i, j] = this_score
            sw_path[i, j] = this_dir
    return sw_score, sw_path


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
                   gap_extension=GAP_EXTENTION):
    """ Smith-Waterman local sequence alignment

    :param seq1:
        A string containing the first sequence, as returned from read_fasta
    :param seq2:
        A string containing the second sequence, as returned from read_fasta
    :param score:
        A pandas DataFrame as returned from read_score_matrix
    :returns:
        Two strings, alignment1, alignment2
        These are the highest scoring **local** alignment of the two
        sequences. In a tie, one of the highest scoring alignments will be
        returned.
    """

    # Calculate the score and path matricies
    sw_score, sw_path = _calc_sw_matrix(seq1, seq2, score,
                                        gap_opening=gap_opening,
                                        gap_extension=gap_extension)

    # Use the score and path to traceback
    align1, align2 = _calc_sw_traceback(seq1, seq2, sw_score, sw_path)
    return align1, align2
