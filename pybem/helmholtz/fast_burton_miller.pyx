#!/usr/bin/env python3
"""
2019-10-25 14:17:45
@author: Paul Reiter
"""
import numpy as np
from scipy.integrate import fixed_quad
from scipy.special import hankel2, struve
import cython
cimport scipy.special.cython_special as cs
# hankel2 = cs.hankel2
# struve = cs.struve


def complex_system_matrix(mesh, *args, **kwargs):
    dimension = len(mesh.elements)
    system_matrix = np.empty((dimension, dimension), dtype=complex)
    for i, diag in enumerate(system_matrix):
        system_matrix[i, i] = main_diagonal(mesh, i, *args, **kwargs)
    rest_of_the_matrix(
        system_matrix,
        system_matrix.shape[0],
        mesh.normals,
        mesh.centers,
        mesh.corners,
        mesh.admittances,
        *args,
        **kwargs
    )
    return system_matrix


w1, w2, w3, w4, w5 = 0.5688888888888889, 0.4786286704993665, 0.4786286704993665, 0.2369268850561891, 0.2369268850561891
x1, x2, x3, x4, x5 = (0.0000000000000000), (-0.5384693101056831), (0.5384693101056831), (-0.9061798459386640), (0.9061798459386640)


def fifth_quad(f1, f2, f3, f4, f5):
    return (
        + w1 * f1
        + w2 * f2
        + w3 * f3
        + w4 * f4
        + w5 * f5
    )


def line_integral(function, p0, p1):
    line_vector = p1 - p0
    length = np.sqrt(line_vector.dot(line_vector))

    def to_quad(t):
        return function((p1 + p0) / 2 + t * (p1 - p0) / 2)

    return length * fifth_quad(
        to_quad(x1),
        to_quad(x2),
        to_quad(x3),
        to_quad(x4),
        to_quad(x5),
    ) / 2


def regularized_hypersingular_bm_part(v):
    return hankel2(1, v) / v - 2j / (np.pi * v ** 2)  # regularization


def main_diagonal(mesh, idx, z0, k, coupling_sign):
    corners, admittance = mesh.corners[idx], mesh.admittances[idx]
    element_vector = corners[1] - corners[0]
    length = np.sqrt(element_vector.dot(element_vector))
    lk2 = length * k / 2
    return (
        +(z0 * admittance * np.pi * length * k / 8)
        * (+hankel2(0, lk2) * struve(-1, lk2) + hankel2(1, lk2) * struve(0, lk2))
        - coupling_sign * fixed_quad(regularized_hypersingular_bm_part, 0, lk2)[0] / 2
        + coupling_sign * 2j / (np.pi * k * length)
        + (1 - coupling_sign * z0 * admittance) / 2
    )


def rest_of_the_matrix(
    complex[:, :] matrix,
    int shape,
    double[:, :] mesh_normals,
    double[:, :] mesh_centers,
    # double[:, :, :] mesh_corners,
    mesh_corners,
    complex[:] mesh_admittances,
    double z0,
    double k,
    int coupling_sign,
):
    for row_idx in range(shape):
        for col_idx in range(shape):
            if row_idx == col_idx:
                continue

            # nx, ny = mesh_normals[row_idx, 0], mesh_normals[row_idx, 1]
            nx = mesh_normals[row_idx, 0]
            ny = mesh_normals[row_idx, 1]
            nsx = mesh_normals[col_idx, 0]
            nsy = mesh_normals[col_idx, 1]
            rx = mesh_centers[row_idx, 0]
            ry = mesh_centers[row_idx, 1]
            admittance = mesh_admittances[col_idx]

            # length = np.sqrt(
            #     + (mesh_corners[col_idx, 1, 0] - mesh_corners[col_idx, 0, 0])**2
            #     + (mesh_corners[col_idx, 1, 1] - mesh_corners[col_idx, 0, 1])**2
            # )
            corners = mesh_corners[col_idx]
            # element_vector = corners[1] - corners[0]
            # length = np.sqrt(element_vector.dot(element_vector))

            # lk2 = length * k / 2

            @cython.cdivision(True)
            def integral_function(double[:] rs):
                cdef double complex adm = admittance
                cdef double vectorx = rx - rs[0]
                cdef double vectory = ry - rs[1]
                cdef double distance = np.sqrt(vectorx**2 + vectory**2)

                cdef double ndv = nx * vectorx + ny * vectory
                cdef double nsdv = nsx * vectorx + nsy * vectory
                cdef double ndns = nx * nsx + ny * nsy

                cdef double kdist = k * distance
                cdef double complex h20 = cs.hankel2(0, kdist)
                cdef double complex h21 = cs.hankel2(1, kdist)
                cdef double complex h22 = cs.hankel2(2, kdist)
                cdef double complex result = (
                    + .25 * k * z0 * adm * h20
                    - coupling_sign * (
                        + 1j * k * nsdv * h21
                        + ndns * h21
                        + z0 * adm * 1j * k * ndv * h21
                        - k * ndv * nsdv * h22 / distance
                     ) / (4. * distance)
                )
                return result

            matrix[row_idx, col_idx] = line_integral(
                integral_function, corners[0], corners[1]
            )


def burton_miller_rhs(mesh, p_inc, grad_p_inc, k, coupling_sign=-1):
    return p_inc + (coupling_sign * 1j / k) * (grad_p_inc * mesh.normals).sum(axis=1)


def fast_burton_miller_solver(
    mesh, p_incoming, grad_p_incoming, z0, k, coupling_sign=-1
):
    matrix = complex_system_matrix(mesh, z0, k, coupling_sign)
    rhs = burton_miller_rhs(mesh, p_incoming, grad_p_incoming, k, coupling_sign)
    return np.linalg.solve(matrix, rhs)
