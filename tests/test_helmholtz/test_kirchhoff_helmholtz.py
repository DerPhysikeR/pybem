#!/usr/bin/env python3
"""
2019-10-25 15:08:08
@author: Paul Reiter
"""
import numpy as np
from scipy.special import hankel2
import pytest
from pybem import (
    Mesh,
    g_2d,
    hs_2d,
    vector_h_2d,
    admitant_2d_integral,
)


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
def test_hs_2d(n, k, r, rs, solution):
    assert solution == hs_2d(n, k, r, rs)


def test_vector_h_2d_is_gradient_of_g_2d():
    k, r, rs = 3, np.array([1, 1]), np.array([0, 0]),
    delta = 1e-8
    delta_x, delta_y = np.array([delta, 0]), np.array([0, delta])
    np.testing.assert_almost_equal(
        vector_h_2d(k, r, rs),
        np.array([
            (g_2d(k, r+delta_x/2, rs) - g_2d(k, r-delta_x/2, rs)),
            (g_2d(k, r+delta_y/2, rs) - g_2d(k, r-delta_y/2, rs))
        ])/delta
    )


def test_admitant_2d_integral():
    mesh = Mesh(np.array([[0, 0], [1, 0]]), np.array([[0, 1]]))
    result = admitant_2d_integral(1, np.array([1, 0]), mesh, 0, 1, 1)
    assert isinstance(result, complex)


def test_admitant_2d_integral_fully_reflective_single_plane():
    mesh = Mesh(np.array([[0, -.5], [0, .5]]), np.array([[0, 1]]))
    result = admitant_2d_integral(1, np.array([0, 1]), mesh, 0, 1, 1)
    np.testing.assert_almost_equal(0, result)
