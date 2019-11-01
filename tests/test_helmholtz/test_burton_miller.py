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
    burton_miller_rhs,
    regularized_hypersingular_bm_part,
)


@pytest.mark.parametrize(
    "k, l",
    [
        # 2*np.pi/(k*elements_per_wavelength)
        (2, 2 * np.pi / (2 * 6)),
        (2, 2 * np.pi / (2 * 10)),
    ],
)
def test_regularized_hypersingular_bm_part_numerical_accuracy(k, l):
    assert (
        complex_relative_error(
            complex_quad(regularized_hypersingular_bm_part, 0, k * l / 2),
            fixed_quad(regularized_hypersingular_bm_part, 0, k * l / 2, n=20)[0],
        )
        < 1e-3
    )


def test_burton_miller_rhs():
    k = 1
    p_inc = np.array([1, 2])
    grad_p_inc = np.array([[2, 3], [2, 3]])
    mesh = types.SimpleNamespace()
    mesh.normals = np.array([[1, 0], [0, 1]])
    np.testing.assert_almost_equal(
        burton_miller_rhs(mesh, p_inc, grad_p_inc, k), np.array([1 - 2j, 2 - 3j])
    )
