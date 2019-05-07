#!/usr/bin/env python3
"""
2019-05-05 12:18:24
@author: Paul Reiter
"""
import numpy as np
from scipy.special import hankel2
import pytest
from integrals import g_2d, h_2d


@pytest.mark.parametrize('k, r, rs, solution', [
    (2, np.array([0, 0]), np.array([1, 0]), 1j*hankel2(0, 2)/4),
    (2, np.array([0, 0]), np.array([3, 4]), 1j*hankel2(0, 10)/4),
])
def test_g_2d(k, r, rs, solution):
    assert solution == g_2d(k, r, rs)


@pytest.mark.parametrize('n, k, r, rs, solution', [
    (np.array([0, 1]), 2, np.array([0, 0]), np.array([1, 0]),
     0),
    (np.array([1, 0]), 2, np.array([0, 0]), np.array([1, 0]),
     1j/4 * (hankel2(-1, 2) - hankel2(1, 2))),
    (np.array([.5, 1]), 2, np.array([0, 0]), np.array([1, 0]),
     1j/8 * (hankel2(-1, 2) - hankel2(1, 2))),
])
def test_h_2d(n, k, r, rs, solution):
    assert solution == h_2d(n, k, r, rs)
