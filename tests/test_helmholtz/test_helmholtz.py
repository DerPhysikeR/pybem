#!/usr/bin/env python3
"""
2019-10-25 17:01:20
@author: Paul Reiter
"""
import numpy as np
import pytest
from pybem import Mesh, calc_solution_at
from pybem.helmholtz import (
    g_2d,
    admitant_2d_integral,
    kirchhoff_helmholtz_solver,
    burton_miller_solver,
    vector_h_2d,
)


@pytest.mark.slow
@pytest.mark.parametrize('solver', [
    kirchhoff_helmholtz_solver,
    burton_miller_solver,
])
def test_calc_solution_at_point_source_reflective_plane(solver):
    # actually solve the linear system for point source above reflective plane
    n = 80
    # create admitant mesh
    mesh = Mesh([(x, 0) for x in np.linspace(6, -6, n+1)],
                [(i, i+1) for i in range(n)])
    k, rho, c = 2*np.pi*300/343, 1, 343
    p_incoming = np.array([g_2d(k, point, np.array([0., 1.]))
                           for point in mesh.centers], dtype=complex)
    grad_p_incoming = np.array([vector_h_2d(k, point, np.array([0., 1.]))
                               for point in mesh.centers], dtype=complex)
    surface_pressure = solver(mesh, p_incoming, grad_p_incoming, k, rho, c)
    solution = calc_solution_at(mesh, admitant_2d_integral, k,
                                          surface_pressure,
                                          np.array([[0., .5]]), rho, c)
    np.testing.assert_allclose(g_2d(k, np.array([0., .5]),
                                    np.array([0., -1.])),
                               solution[0], rtol=1e-2)


@pytest.mark.slow
@pytest.mark.parametrize('solver', [
    kirchhoff_helmholtz_solver,
    burton_miller_solver,
])
def test_calc_reflection_of_fully_absorbing_plane_for_plane_wave(solver):
    # plane wave impinging normally on admittance plane
    n = 180
    # create admitant mesh
    k, rho, c = 2*np.pi*300/343, 1, 343
    mesh = Mesh([(x, 0) for x in np.linspace(9, -9, n+1)],
                [(i, i+1) for i in range(n)], [1/343. for _ in range(n)])
    p_incoming = np.array([np.exp(1j*k*point[1])
                           for point in mesh.centers], dtype=complex)
    grad_p_incoming = np.array([[1j*k*np.exp(1j*k*point[1]), 0]
                                for point in mesh.centers],
                               dtype=complex)
    surface_pressure = solver(mesh, p_incoming, grad_p_incoming, k, rho, c)
    solution = calc_solution_at(mesh, admitant_2d_integral, k,
                                          surface_pressure,
                                          np.array([[0., .5]]), rho, c)
    np.testing.assert_allclose(0+1, solution[0]+1, rtol=1e-2)
