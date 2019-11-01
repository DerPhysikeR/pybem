#!/usr/bin/env python3
"""
2019-10-25 14:17:02
@author: Paul Reiter
"""
import numpy as np
from .helmholtz import g_2d, hs_2d
from ..pybem import complex_system_matrix
from ..integrals import line_integral


def admitant_2d_matrix_element(mesh, row_idx, col_idx, z0, k):
    ns, r = mesh.normals[col_idx], mesh.centers[row_idx]
    corners, admittance = mesh.corners[col_idx], mesh.admittances[col_idx]
    singular = row_idx == col_idx

    def integral_function(rs):
        return hs_2d(ns, k, r, rs) - 1j * k * z0 * admittance * g_2d(k, r, rs)

    return (
        line_integral(integral_function, corners[0], corners[1], singular)
        + singular / 2
    )


def kirchhoff_helmholtz_solver(mesh, p_incoming, z0, k):
    matrix = complex_system_matrix(admitant_2d_matrix_element, mesh, z0, k)
    return np.linalg.solve(matrix, p_incoming)
