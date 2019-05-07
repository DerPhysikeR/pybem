#!/usr/bin/env python3
"""
2019-05-07 13:21:02
@author: Paul Reiter
"""
import numpy as np
from mesh import Mesh
from pybem import complex_system_matrix


def test_complex_system_matrix():
    mesh = Mesh([[0, 0], [1, 0], [1, 2]], [[0, 1], [1, 2]])

    def integral_function(k, r, admittance, n, corners, singular):
        return r[0] + 1j*r[1] + singular

    reference_system_matrix = np.array([[1.5, .5], [1 + 1j, 2 + 1j]],
                                       dtype=complex)
    system_matrix = complex_system_matrix(mesh, integral_function, 1)
    np.testing.assert_allclose(reference_system_matrix, system_matrix)
