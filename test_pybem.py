#!/usr/bin/env python3
"""
2019-05-07 13:21:02
@author: Paul Reiter
"""
import numpy as np
import pytest
from mesh import Mesh
from pybem import complex_system_matrix, calc_scattered_pressure_at
from integrals import g_2d, admitant_2d_integral


def test_complex_system_matrix():
    mesh = Mesh([[0, 0], [1, 0], [1, 2]], [[0, 1], [1, 2]])

    def integral_function(k, r, admittance, n, corners, singular):
        return r[0] + 1j*r[1] + singular

    reference_system_matrix = np.array([[1.5, .5], [1 + 1j, 2 + 1j]],
                                       dtype=complex)
    system_matrix = complex_system_matrix(mesh, integral_function, 1)
    np.testing.assert_allclose(reference_system_matrix, system_matrix)


def test_calc_scattered_pressure_at():
    # point source above fully reflective plane
    n = 200
    mesh = Mesh([(x, 0) for x in np.linspace(10, -10, n+1)],
                [(i, i+1) for i in range(n)])
    k, rho, omega = 2*np.pi*300/343, 1, 2*np.pi*300
    surface_pressure = 2*np.array([g_2d(k, point, np.array([0., 1.]))
                                   for point in mesh.centers], dtype=complex)
    solution = calc_scattered_pressure_at(mesh, admitant_2d_integral, k,
                                          surface_pressure,
                                          np.array([[0., .5]]), rho, omega)
    np.testing.assert_allclose(g_2d(k, np.array([0., .5]),
                                    np.array([0., -1.])),
                               solution[0], rtol=1e-3)


@pytest.mark.slow
def test_calc_scattered_pressure_at_point_source_reflective_plane():
    # actually solve the linear system for point source above reflective plane
    n = 80
    # create admitant mesh
    mesh = Mesh([(x, 0) for x in np.linspace(4, -4, n+1)],
                [(i, i+1) for i in range(n)])
    k, rho, omega = 2*np.pi*300/343, 1, 2*np.pi*300
    system_matrix = complex_system_matrix(mesh, admitant_2d_integral, k, rho,
                                          omega)
    p_incoming = np.array([g_2d(k, point, np.array([0., 1.]))
                           for point in mesh.centers], dtype=complex)
    surface_pressure = np.linalg.solve(system_matrix, -p_incoming)
    solution = calc_scattered_pressure_at(mesh, admitant_2d_integral, k,
                                          surface_pressure,
                                          np.array([[0., .5]]), rho, omega)
    np.testing.assert_allclose(g_2d(k, np.array([0., .5]),
                                    np.array([0., -1.])),
                               solution[0], rtol=1e-2)
