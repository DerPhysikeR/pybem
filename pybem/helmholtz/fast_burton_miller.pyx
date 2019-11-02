#!/usr/bin/env python3
"""
2019-10-25 14:17:45
@author: Paul Reiter
"""
import numpy as np
import cython
from scipy.special.cython_special cimport hankel2 as chankel2
from scipy.special.cython_special cimport struve as cstruve
from libc.math cimport sqrt, pi
from .burton_miller import burton_miller_rhs


WEIGHTS = np.array([0.5688888888888889, 0.4786286704993665, 0.4786286704993665, 0.2369268850561891, 0.2369268850561891])
ABSCISSA = np.array([0.0000000000000000, -0.5384693101056831, 0.5384693101056831, -0.9061798459386640, 0.9061798459386640])


def fast_burton_miller_solver(
    mesh, p_incoming, grad_p_incoming, z0, k, coupling_sign=-1
):
    matrix = complex_system_matrix(mesh, z0, k, coupling_sign)
    rhs = burton_miller_rhs(mesh, p_incoming, grad_p_incoming, k, coupling_sign)
    return np.linalg.solve(matrix, rhs)


def complex_system_matrix(mesh, *args, **kwargs):
    dimension = len(mesh.elements)
    system_matrix = np.empty((dimension, dimension), dtype=complex)
    c_calc_system_matrix(
        system_matrix,
        system_matrix.shape[0],
        mesh.normals,
        mesh.centers,
        mesh.corners,
        mesh.admittances,
        args[0],
        args[1],
        args[2],
        WEIGHTS,
        ABSCISSA,
        len(WEIGHTS),
    )
    return system_matrix


@cython.cdivision(True)
cdef double complex  regularized_hypersingular_bm_part(double v):
    return chankel2(1, v) / v - 2j / (pi * v ** 2)  # regularization


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void c_calc_system_matrix(
    complex[:, :] matrix,
    int shape,
    double[:, :] mesh_normals,
    double[:, :] mesh_centers,
    double[:, :, :] mesh_corners,
    complex[:] mesh_admittances,
    double z0,
    double k,
    int coupling_sign,
    double[:] weights,
    double[:] abscissa,
    int n_weights,
):

    cdef:
        double length, lk2, x
        double *p0
        double *p1
        double *r
        double *n
        double *ns
        double rs[2]
        cdef double element_vector[2]
        double complex admittance, result

    for row_idx in range(shape):
        for col_idx in range(shape):

            p0, p1 = &mesh_corners[col_idx, 0, 0], &mesh_corners[col_idx, 1, 0]
            admittance = mesh_admittances[col_idx]
            element_vector[:] = [p1[0] - p0[0], p1[1] - p0[1]]
            length = sqrt(inner_product(element_vector, element_vector))

            if row_idx == col_idx:
                lk2 = length * k / 2

                # perform integration
                result = 0
                for i in range(n_weights):
                    x = lk2/2 + abscissa[i] * lk2/2
                    result = result + weights[i] * regularized_hypersingular_bm_part(x)

                matrix[row_idx, col_idx] = (
                    +(z0 * admittance * pi * length * k / 8)
                    * (+chankel2(0, lk2) * cstruve(-1, lk2) + chankel2(1, lk2) * cstruve(0, lk2))
                    + coupling_sign * 2j / (pi * k * length)
                    + (1 - coupling_sign * z0 * admittance) / 2
                    - coupling_sign * lk2 * result / 4
                )

            else:

                n, ns = &mesh_normals[row_idx, 0], &mesh_normals[col_idx, 0]
                r = &mesh_centers[row_idx, 0]

                # perform integration
                result = 0
                for i in range(n_weights):
                    rs[0] = (p1[0] + p0[0]) / 2 + abscissa[i] * (p1[0] - p0[0]) / 2
                    rs[1] = (p1[1] + p0[1]) / 2 + abscissa[i] * (p1[1] - p0[1]) / 2
                    result = (
                        result + weights[i] * rest_integral_function(
                            r, n, admittance, k, z0, coupling_sign, rs, ns
                        )
                    )
                matrix[row_idx, col_idx] = .5 * length * result


cdef double inner_product(double *vec1, double *vec2):
    return vec1[0] * vec2[0] + vec1[1] * vec2[1]


@cython.cdivision(True)
cdef double complex rest_integral_function(
    double *r,
    double *n,
    double complex adm,
    double k,
    double z0,
    int coupling_sign,
    double *rs,
    double *ns,
):
    cdef:
        double difference[2]
    difference[:] = [r[0] - rs[0], r[1] - rs[1]]
    cdef:
        double distance = sqrt(inner_product(difference, difference))
        double ndv = inner_product(n, difference)
        double nsdv = inner_product(ns, difference)
        double ndns = inner_product(n, ns)
        double kdist = k * distance
        double complex h20 = chankel2(0, kdist)
        double complex h21 = chankel2(1, kdist)
        double complex h22 = chankel2(2, kdist)
        double complex result = (
            + .25 * k * z0 * adm * h20
            - coupling_sign * (
                + 1j * k * nsdv * h21
                + ndns * h21
                + z0 * adm * 1j * k * ndv * h21
                - k * ndv * nsdv * h22 / distance
                ) / (4. * distance)
        )
    return result
