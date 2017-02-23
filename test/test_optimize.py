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


def test_calc_distribution():
    """ Tests for the distribution calculator """

    # Default score is 0
    score = optimize.calc_distribution([])

    exp_score = np.zeros((24, 24))

    np.testing.assert_almost_equal(score.values, exp_score)

    # Score for enrichment in AA pairs
    score = optimize.calc_distribution([
        ('AAAAA', 'AAAAA'),
        ('AAAA', 'AAAA'),
        ('AAAAAA'*100, 'AAAAAA'*100),
    ])

    assert np.allclose(score.loc['A', 'A'], 10.25, atol=1e-2)

    # Score for enrichment in CS pairs
    score = optimize.calc_distribution([
        ('CSCSCS', 'SCSCSC'),
        ('SCCSSS', 'SSCCCS'),
    ])

    # Make sure it's symmetric
    assert np.allclose(score, score.T)

    # Check specific cells
    assert np.allclose(score.loc['S', 'S'], 2.32, atol=1e-2)
    assert np.allclose(score.loc['C', 'C'], 1.58, atol=1e-2)
    assert np.allclose(score.loc['S', 'C'], 3.32, atol=1e-2)
    assert np.allclose(score.loc['C', 'S'], 3.32, atol=1e-2)

    # Score for enrichment in gaps
    score = optimize.calc_distribution([
        ('C----S', 'SCSCSC'),
        ('SCCS-S', 'S----S'),
    ])

    # Make sure it's still symmetric
    assert np.allclose(score, score.T)

    # Check specific cells
    assert np.allclose(score.loc['*', '*'], 1.58, atol=1e-2)
    assert np.allclose(score.loc['*', 'C'], 2.32, atol=1e-2)
    assert np.allclose(score.loc['C', '*'], 2.32, atol=1e-2)
    assert np.allclose(score.loc['S', '*'], 2.0, atol=1e-2)
