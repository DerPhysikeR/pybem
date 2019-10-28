#!/usr/bin/env python3
"""
2019-10-25 14:17:02
@author: Paul Reiter
"""
import numpy as np
from scipy.special import hankel2
from ..pybem import complex_system_matrix
from ..integrals import line_integral


def admitant_2d_integral(mesh, idx, point, k, z0):
    n = mesh.normals[idx]
    admittance = mesh.admittances[idx]
    corners = mesh.corners[idx]

    def integral_function(rs):
        return -hs_2d(n, k, point, rs) + 1j * k * z0 * admittance * g_2d(k, point, rs)

    return line_integral(integral_function, corners[0], corners[1], False)


def admitant_2d_matrix_element(mesh, row_idx, col_idx, k, z0):
    ns, r = mesh.normals[col_idx], mesh.centers[row_idx]
    corners, admittance = mesh.corners[col_idx], mesh.admittances[col_idx]
    singular = row_idx == col_idx

    def integral_function(rs):
        return hs_2d(ns, k, r, rs) - 1j * k * z0 * admittance * g_2d(k, r, rs)

    return (
        line_integral(integral_function, corners[0], corners[1], singular)
        + singular / 2
    )


def vector_h_2d(k, r, rs):
    """Vectorial gradient of the 2D Green's function acoording to the obverver
       point r"""
    distance = np.sqrt((r - rs).dot(r - rs))
    return -1j * k * (r - rs) / (4 * distance) * hankel2(1, k * distance)


def hs_2d(ns, k, r, rs):
    """Gradient of the 2D Green's function according to the source point rs"""
    return -ns.dot(vector_h_2d(k, r, rs))


def g_2d(k, r, rs):
    """2D Green's function"""
    return 1j * hankel2(0, k * np.sqrt((r - rs).dot(r - rs))) / 4


def kirchhoff_helmholtz_solver(mesh, p_incoming, grad_p_incoming, k, z0):
    matrix = complex_system_matrix(admitant_2d_matrix_element, mesh, k, z0)
    return np.linalg.solve(matrix, p_incoming)
