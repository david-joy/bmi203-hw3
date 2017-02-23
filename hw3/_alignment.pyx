
import numpy as np
cimport numpy as np


def _encode_sw_matrix(str seq1,
                      str seq2,
                      score):
    # Convert the python strings to numpy arrays

    # This allows `_calc_sw_matrix` to operate in pure C-mode
    # which is something like 4x faster with arrays
    cdef np.ndarray enc_seq1 = np.zeros((len(seq1), ), dtype=np.int)
    cdef np.ndarray enc_seq2 = np.zeros((len(seq2), ), dtype=np.int)

    cdef str ci
    cdef int i

    for i, ci in enumerate(seq1):
        enc_seq1[i] = score.columns.get_loc(ci)

    for i, ci in enumerate(seq2):
        enc_seq2[i] = score.index.get_loc(ci)

    return enc_seq1, enc_seq2, score.values


def _calc_sw_traceback(str seq1,
                       str seq2,
                       np.ndarray sw_score,
                       np.ndarray sw_path):
    # Calculate the traceback for Smith-Waterman

    cdef int idx_max, i, j
    cdef int this_dir

    cdef float score

    cdef list align1, align2
    cdef int rows = sw_score.shape[0]
    cdef int cols = sw_score.shape[1]

    cdef str align1_str, align2_str

    # Find the best score in the matrix
    idx_max = np.argmax(sw_score)
    i, j = np.unravel_index(idx_max, (rows, cols))

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
    align1_str = ''.join(reversed(align1))
    align2_str = ''.join(reversed(align2))
    return align1_str, align2_str


def _calc_sw_matrix(np.ndarray enc_seq1,
                    np.ndarray enc_seq2,
                    np.ndarray enc_score,
                    float gap_opening,
                    float gap_extension):
    # Calculate the score matrix for Smith-Waterman

    cdef int i, j
    cdef float penalty_up, penalty_left
    cdef float prev_dir_up, prev_dir_left
    cdef float diag, up, left
    cdef float this_score, this_dir

    cdef int ci, cj

    # Initialize a score matrix with rows for seq2, cols for seq1
    cdef int rows = enc_seq2.shape[0]
    cdef int cols = enc_seq1.shape[0]
    cdef np.ndarray sw_score = np.zeros((rows + 1, cols + 1), dtype=np.float32)
    cdef np.ndarray sw_path = np.zeros((rows + 1, cols + 1), dtype=np.float32)

    # Scan through the matrix from top-left to bottom-right
    for i in range(1, rows+1):
        ci = enc_seq2[i-1]
        for j in range(1, cols+1):
            cj = enc_seq1[j-1]

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
            diag = sw_score[i-1, j-1] + enc_score[ci, cj]
            up = sw_score[i-1, j] + penalty_up
            left = sw_score[i, j-1] + penalty_left

            # The new direction is 0 - diagonal, 1 - up, 2 - left
            this_dir = np.argmax([diag, up, left])
            this_score = np.max([diag, up, left, 0])

            sw_score[i, j] = this_score
            sw_path[i, j] = this_dir
    return sw_score, sw_path
