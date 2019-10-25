#!/usr/bin/env python3
"""
2019-10-25 15:11:37
@author: Paul Reiter
"""
import types
import numpy as np
from scipy.integrate import fixed_quad
import pytest
from pybem import complex_relative_error, complex_quad
from pybem.helmholtz.burton_miller import (
    g_2d,
    h_2d,
    hs_2d,
    hypersingular,
    burton_miller_rhs,
    regularized_hypersingular_bm_part,
)


def test_h_2d_is_gradient_of_g_2d():
    k, r, rs, n = 3, np.array([2, 0]), np.array([0, 0]), np.array([1, 0])
    delta = np.array([1e-8, 0])
    np.testing.assert_almost_equal(
        h_2d(n, k, r, rs),
        (g_2d(k, r+delta/2, rs) - g_2d(k, r-delta/2, rs)) / delta[0]
    )


def test_hypersingular_is_scalar():
    with pytest.raises(TypeError):
        len(hypersingular(1, np.array([2, 3]), np.array([3, 2]),
                          np.array([1, 0]), np.array([0, 1])))


def test_hypersingular_is_complex():
    assert isinstance(hypersingular(1, np.array([2, 3]), np.array([3, 2]),
                                    np.array([1, 0]), np.array([0, 1])),
                      np.complex128)


def test_hypersingular_is_equal_after_swap_of_coordinates():
    assert (
        hypersingular(1, np.array([2, 3]), np.array([3, 2]),
                      np.array([1, 0]), np.array([0, 1]))
        ==
        hypersingular(1, np.array([3, 2]), np.array([2, 3]),
                      np.array([1, 0]), np.array([0, 1]))
        )


def test_burton_miller_rhs():
    k = 1
    p_inc = np.array([1, 2])
    grad_p_inc = np.array([[2, 3], [2, 3]])
    mesh = types.SimpleNamespace()
    mesh.normals = np.array([[1, 0], [0, 1]])
    np.testing.assert_almost_equal(
        burton_miller_rhs(k, mesh, p_inc, grad_p_inc),
        np.array([1-2j, 2-3j])
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
        hypersingular(k, r, rs, n, ns),
        (hs_2d(ns, k, r+delta_n, rs) - hs_2d(ns, k, r-delta_n, rs))/delta
    )


@pytest.mark.parametrize('k, l', [
    # 2*np.pi/(k*elements_per_wavelength)
    (2, 2*np.pi/(2*6)),
    (2, 2*np.pi/(2*10)),
])
def test_regularized_hypersingular_bm_part_numerical_accuracy(k, l):
    assert complex_relative_error(
        complex_quad(regularized_hypersingular_bm_part, 0, k*l/2),
        fixed_quad(regularized_hypersingular_bm_part, 0, k*l/2, n=20)[0],
    ) < 1e-3
