
import numpy as np
cimport numpy as np


def _calc_sw_matrix(str seq1,
                    str seq2,
                    score,
                    float gap_opening,
                    float gap_extension):
    # Calculate the score matrix for Smith-Waterman

    cdef int i, j
    cdef float penalty_up, penalty_left
    cdef float prev_dir_up, prev_dir_left
    cdef float diag, up, left
    cdef float this_score, this_dir

    cdef str ci, cj

    # Initialize a score matrix with rows for seq2, cols for seq1
    cdef int rows = len(seq2)
    cdef int cols = len(seq1)
    cdef np.ndarray sw_score = np.zeros((rows + 1, cols + 1), dtype=np.float32)
    cdef np.ndarray sw_path = np.zeros((rows + 1, cols + 1), dtype=np.float32)

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
