#!/usr/bin/env python3
"""
2019-05-07 13:21:02
@author: Paul Reiter
"""
import numpy as np
from pybem import __version__
from pybem import Mesh
from pybem.helmholtz import g_2d, admitant_2d_integral
from pybem.pybem import (
    complex_system_matrix,
    calc_solution_at,
)


def test_version():
    assert __version__ == '0.3.0'


def test_complex_system_matrix():
    mesh = Mesh([[0, 0], [1, 0], [1, 2]], [[0, 1], [1, 2]])

    # def integral_function(k, r, admittance, n, corners, singular):
    def integral_function(k, mesh, row_idx, col_idx):
        return (mesh.centers[row_idx][0] + 1j*mesh.centers[row_idx][1]
                + (row_idx == col_idx))

    reference_system_matrix = np.array([[1.5, .5], [1 + 1j, 2 + 1j]],
                                       dtype=complex)
    system_matrix = complex_system_matrix(mesh, integral_function, 1)
    np.testing.assert_allclose(reference_system_matrix, system_matrix)


def test_solution_at():
    # point source above fully reflective plane
    n = 200
    mesh = Mesh([(x, 0) for x in np.linspace(10, -10, n+1)],
                [(i, i+1) for i in range(n)])
    k, rho, c = 2*np.pi*300/343, 1, 343
    surface_pressure = 2*np.array([g_2d(k, point, np.array([0., 1.]))
                                   for point in mesh.centers], dtype=complex)
    solution = calc_solution_at(mesh, admitant_2d_integral, k,
                                          surface_pressure,
                                          np.array([[0., .5]]), rho, c)
    np.testing.assert_allclose(g_2d(k, np.array([0., .5]),
                                    np.array([0., -1.])),
                               solution[0], rtol=1e-3)
