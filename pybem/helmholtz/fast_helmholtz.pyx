#!/usr/bin/env python3
"""
2019-11-02 09:14:34
@author: Paul Reiter
"""
import numpy as np
from ..integrals import line_integral
from .helmholtz import g_2d, hs_2d


def fast_calc_solution_at(
        integral_function, mesh, surface_solution, points_of_interest, *args, **kwargs
):
    assert len(mesh.elements) == len(surface_solution)
    points_of_interest = np.array(points_of_interest, dtype=float)
    solution = np.zeros(len(points_of_interest), dtype=complex)
    for i, point in enumerate(points_of_interest):
        for j, sp in enumerate(surface_solution):
            solution[i] += sp * integral_function(mesh, j, point, *args, **kwargs)
    return solution


def admitant_2d_integral(mesh, idx, point, z0, k):
    n = mesh.normals[idx]
    admittance = mesh.admittances[idx]
    corners = mesh.corners[idx]

    def integral_function(rs):
        return -hs_2d(n, k, point, rs) + 1j * k * z0 * admittance * g_2d(k, point, rs)

    return line_integral(integral_function, corners[0], corners[1], False)
