#!/usr/bin/env python3
"""
2019-10-25 14:17:45
@author: Paul Reiter
"""
import numpy as np
from scipy.integrate import fixed_quad
from scipy.special import hankel2, struve


def complex_system_matrix(matrix_element_function, mesh, *args, **kwargs):
    dimension = len(mesh.elements)
    system_matrix = np.empty((dimension, dimension), dtype=complex)
    for i, row in enumerate(system_matrix):
        for j, col in enumerate(row):
            system_matrix[i, j] = matrix_element_function(mesh, i, j, *args, **kwargs)
    return system_matrix


def line_integral(function, p0, p1):
    line_vector = p1 - p0
    length = np.sqrt(line_vector.dot(line_vector))

    def to_quad(t):
        return function((p1 + p0) / 2 + t * (p1 - p0) / 2)

    return length * fixed_quad(np.vectorize(to_quad), -1.0, 1.0)[0] / 2


def regularized_hypersingular_bm_part(v):
    return hankel2(1, v) / v - 2j / (np.pi * v ** 2)  # regularization


def admitant_2d_matrix_element_bm(mesh, row_idx, col_idx, z0, k, coupling_sign):
    n, ns = mesh.normals[row_idx], mesh.normals[col_idx]
    r = mesh.centers[row_idx]
    corners, admittance = mesh.corners[col_idx], mesh.admittances[col_idx]
    singular = row_idx == col_idx

    if singular:
        element_vector = corners[1] - corners[0]
        length = np.sqrt(element_vector.dot(element_vector))
        lk2 = length * k / 2
        return (
            +(z0 * admittance * np.pi * length * k / 8)
            * (+hankel2(0, lk2) * struve(-1, lk2) + hankel2(1, lk2) * struve(0, lk2))
            - coupling_sign
            * fixed_quad(regularized_hypersingular_bm_part, 0, lk2)[0]
            / 2
            + coupling_sign * 2j / (np.pi * k * length)
            + (1 - coupling_sign * z0 * admittance) / 2
        )

    else:

        def integral_function(rs):
            return (
                hs_2d(ns, k, r, rs)
                - 1j * k * z0 * admittance * g_2d(k, r, rs)
                + coupling_sign * z0 * admittance * h_2d(n, k, r, rs)
                + coupling_sign * 1j / k * hypersingular(k, r, rs, n, ns)
            )

        return line_integral(integral_function, corners[0], corners[1])


def hypersingular(k, r, rs, n, ns):
    vector = r - rs
    distance = np.sqrt(vector.dot(vector))
    kdist = k * distance
    return (1j * k / (4 * distance ** 2)) * (
        distance * hankel2(1, kdist) * n.dot(ns)
        - k * hankel2(2, kdist) * n.dot(vector) * ns.dot(vector)
    )


def burton_miller_rhs(mesh, p_inc, grad_p_inc, k, coupling_sign=-1):
    return p_inc + (coupling_sign * 1j / k) * (grad_p_inc * mesh.normals).sum(axis=1)


def h_2d(n, k, r, rs):
    """Gradient of the 2D Green's function according to the obverver point r"""
    return -hs_2d(n, k, r, rs)


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


def fast_burton_miller_solver(mesh, p_incoming, grad_p_incoming, z0, k, coupling_sign=-1):
    matrix = complex_system_matrix(
        admitant_2d_matrix_element_bm, mesh, z0, k, coupling_sign
    )
    rhs = burton_miller_rhs(mesh, p_incoming, grad_p_incoming, k, coupling_sign)
    return np.linalg.solve(matrix, rhs)
