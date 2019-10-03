#!/usr/bin/env python3
"""
2019-05-05 12:18:24
@author: Paul Reiter
"""
import numpy as np
from scipy.special import hankel2
import pytest
import pybem as pb


@pytest.mark.parametrize('k, r, rs, solution', [
    (2, np.array([0, 0]), np.array([1, 0]), 1j*hankel2(0, 2)/4),
    (2, np.array([0, 0]), np.array([3, 4]), 1j*hankel2(0, 10)/4),
])
def test_g_2d(k, r, rs, solution):
    assert solution == pb.g_2d(k, r, rs)


@pytest.mark.parametrize('n, k, r, rs, solution', [
    (np.array([0, 1]), 2, np.array([0, 0]), np.array([1, 0]),
     0),
    (np.array([1, 0]), 2, np.array([0, 0]), np.array([1, 0]),
     1j/4 * (hankel2(-1, 2) - hankel2(1, 2))),
    (np.array([.5, 1]), 2, np.array([0, 0]), np.array([1, 0]),
     1j/8 * (hankel2(-1, 2) - hankel2(1, 2))),
])
def test_hs_2d(n, k, r, rs, solution):
    assert solution == pb.hs_2d(n, k, r, rs)


def test_line_integral():

    def square(coords):
        return coords.dot(coords)

    result = pb.line_integral(square, [0, 0], [3, 4], False)
    np.testing.assert_almost_equal(125/3, result)


def test_line_integral_singular():

    def singular_function(coords):
        return 1/coords[0]

    result = pb.line_integral(singular_function, [-1, 0], [1, 0], True)
    np.testing.assert_almost_equal(0, result)


def test_admitant_2d_integral():
    mesh = pb.Mesh(np.array([[0, 0], [1, 0]]), np.array([[0, 1]]))
    result = pb.admitant_2d_integral(1, np.array([1, 0]), mesh, 0, 1, 1)
    assert isinstance(result, complex)


def test_admitant_2d_integral_fully_reflective_single_plane():
    mesh = pb.Mesh(np.array([[0, -.5], [0, .5]]), np.array([[0, 1]]))
    result = pb.admitant_2d_integral(1, np.array([0, 1]), mesh, 0, 1, 1)
    np.testing.assert_almost_equal(0, result)
