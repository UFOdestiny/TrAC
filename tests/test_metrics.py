import numpy as np

from trac.metrics import aurc, brier, ece, risk_at_coverage


def test_perfect_ranking():
    conf, y = np.array([0.9, 0.8, 0.2, 0.1]), np.array([1, 1, 0, 0])
    assert risk_at_coverage(conf, y, 0.5) == 0.0
    assert aurc(conf, y) < aurc(1 - conf, y)


def test_ties_resolved_in_expectation():
    conf, y = np.ones(4), np.array([1, 0, 1, 0])
    assert risk_at_coverage(conf, y, 0.25) == 0.5
    assert aurc(conf, y) == 0.5


def test_calibration():
    y = np.array([1, 0, 1, 0])
    assert ece(y.astype(float), y) == 0.0
    assert brier(np.full(4, 0.5), y) == 0.25
