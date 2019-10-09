#!/usr/bin/env python3
"""
2019-05-05 12:18:24
@author: Paul Reiter
"""
import types
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


def test_h_2d_is_gradient_of_g_2d():
    k, r, rs, n = 3, np.array([2, 0]), np.array([0, 0]), np.array([1, 0])
    delta = np.array([1e-8, 0])
    np.testing.assert_almost_equal(
        pb.h_2d(n, k, r, rs),
        (pb.g_2d(k, r+delta/2, rs) - pb.g_2d(k, r-delta/2, rs)) / delta[0]
    )


def test_vector_h_2d_is_gradient_of_g_2d():
    k, r, rs = 3, np.array([1, 1]), np.array([0, 0]),
    delta = 1e-8
    delta_x, delta_y = np.array([delta, 0]), np.array([0, delta])
    np.testing.assert_almost_equal(
        pb.vector_h_2d(k, r, rs),
        np.array([
            (pb.g_2d(k, r+delta_x/2, rs) - pb.g_2d(k, r-delta_x/2, rs)),
            (pb.g_2d(k, r+delta_y/2, rs) - pb.g_2d(k, r-delta_y/2, rs))
        ])/delta
    )


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


def test_hypersingular_is_scalar():
    with pytest.raises(TypeError):
        len(pb.hypersingular(1, np.array([2, 3]), np.array([3, 2]),
                             np.array([1, 0]), np.array([0, 1])))


def test_hypersingular_is_complex():
    assert isinstance(pb.hypersingular(1, np.array([2, 3]), np.array([3, 2]),
                                       np.array([1, 0]), np.array([0, 1])),
                      np.complex128)


def test_hypersingular_is_equal_after_swap_of_coordinates():
    assert (
        pb.hypersingular(1, np.array([2, 3]), np.array([3, 2]),
                         np.array([1, 0]), np.array([0, 1]))
        ==
        pb.hypersingular(1, np.array([3, 2]), np.array([2, 3]),
                         np.array([1, 0]), np.array([0, 1]))
        )


def test_burton_miller_rhs():
    k = 1
    p_inc = np.array([1, 2])
    grad_p_inc = np.array([[2, 2], [3, 3]])
    mesh = types.SimpleNamespace()
    mesh.normals = np.array([[1, 0], [0, 1]])
    np.testing.assert_almost_equal(
        pb.burton_miller_rhs(k, mesh, p_inc, grad_p_inc),
        np.array([1+2j, 2+3j])
    )


@pytest.mark.parametrize('k, r, n', [
    (2, np.array([2, 1]), np.array([-1, 0])),
    (1, np.array([2, 1]), np.array([-1, 0])),
    (1, np.array([2, 0]), np.array([0, 1])),
    (1, np.array([2, 0]), np.array([0, -1])),
    (5, np.array([0, 5]), np.array([0, 1])),
])
def test_hypersingular_is_gradient_of_h(k, r, n):
    rs, ns = np.array([0, 0]), np.array([0, 1])
    delta = 1e-8
    delta_n = n * delta / 2
    np.testing.assert_almost_equal(
        pb.hypersingular(k, r, rs, n, ns),
        (pb.hs_2d(ns, k, r+delta_n, rs) - pb.hs_2d(ns, k, r-delta_n, rs))/delta
    )
