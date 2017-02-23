""" Tests for the optimizer module """

# Imports

import numpy as np

from hw3 import optimize

# Tests


def test_score_matrix_objective():
    """ Tests for the score objective function """

    # Test for a really bad objective
    tp = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    fp = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    res = optimize.score_matrix_objective(tp, fp)

    assert res == 0.0

    # Test for a perfect objective
    tp = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    fp = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    res = optimize.score_matrix_objective(tp, fp)

    assert res == 4.0

    # Tests for an objective that is half right
    tp = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    fp = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    res = optimize.score_matrix_objective(tp, fp)

    assert res == 2.0

    # Test for objective where the two are equal lines
    tp = np.arange(0, 10)
    fp = np.arange(0, 10)

    res = optimize.score_matrix_objective(tp, fp)

    assert res == 0.7
