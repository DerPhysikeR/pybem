#!/usr/bin/env python3
"""
2019-11-02 09:14:34
@author: Paul Reiter
"""
import numpy as np
import cython
from scipy.special.cython_special cimport hankel2
from libc.math cimport sqrt


WEIGHTS = np.array([0.5688888888888889, 0.4786286704993665, 0.4786286704993665, 0.2369268850561891, 0.2369268850561891])
ABSCISSA = np.array([0.0000000000000000, -0.5384693101056831, 0.5384693101056831, -0.9061798459386640, 0.9061798459386640])


def fast_calc_solution_at(
    integral_function, mesh, surface_solution, points_of_interest, *args, **kwargs
):
    assert len(mesh.elements) == len(surface_solution)
    points_of_interest = np.array(points_of_interest, dtype=float)
    solution = np.zeros(len(points_of_interest), dtype=complex)
    c_calc_solution(solution, mesh.normals, mesh.corners, mesh.admittances, len(mesh.elements), points_of_interest, len(points_of_interest), surface_solution, args[0], args[1], WEIGHTS, ABSCISSA, len(WEIGHTS))
    return solution


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void c_calc_solution(
        double complex[:] solution,
        double[:, :] mesh_normals,
        double[:, :, :] mesh_corners,
        double complex[:] mesh_admittances,
        int mesh_shape,
        double[:, :] points_of_interest,
        int poi_shape,
        double complex[:] surface_solution,
        double z0,
        double k,
        double[:] weights,
        double[:] abscissa,
        int len_weights
):
    cdef:
        double *element_normal
        double *p0
        double *p1
        double complex element_admittance
        int poi_idx, mesh_idx
        double *point

    for poi_idx in range(poi_shape):
        point = &points_of_interest[poi_idx, 0]
        for mesh_idx in range(mesh_shape):
            element_normal = &mesh_normals[mesh_idx, 0]
            # element_corners[:] = mesh_corners[mesh_idx, 0]
            p0, p1 = &mesh_corners[mesh_idx, 0, 0], &mesh_corners[mesh_idx, 1, 0]
            element_admittance = mesh_admittances[mesh_idx]
            solution[poi_idx] = (
                solution[poi_idx]
                + surface_solution[mesh_idx]
                * element_integral(element_normal, p0, p1, element_admittance, point, z0, k, weights, abscissa, len_weights)
            )


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double complex element_integral(
        double *element_normal,
        double *p0,
        double *p1,
        double complex element_admittance,
        double *point,
        double z0,
        double k,
        double[:] weights,
        double[:] abscissa,
        int len_weights
):
    cdef:
        double element_vector[2]
    element_vector = [p1[0] - p0[0], p1[1] - p0[1]]
    cdef:
        double length = sqrt(inner_product(element_vector, element_vector))
        double complex result
        double rs[2]

    # perform integration
    result = 0
    for i in range(len_weights):
        rs[0] = (p1[0] + p0[0]) / 2 + abscissa[i] * (p1[0] - p0[0]) / 2
        rs[1] = (p1[1] + p0[1]) / 2 + abscissa[i] * (p1[1] - p0[1]) / 2
        result = result + weights[i] * integral_function(point, rs, element_normal, element_admittance, z0, k)

    return .5 * length * result


cdef double inner_product(double *vec1, double *vec2):
    return vec1[0] * vec2[0] + vec1[1] * vec2[1]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double complex integral_function(
        double *point,
        double *rs,
        double *ns,
        double complex admittance,
        double z0,
        double k
):
    cdef:
        double difference[2]
        double distance
        double kdist
        double factor
    difference = [point[0] - rs[0], point[1] - rs[1]]
    distance = sqrt(inner_product(difference, difference))
    kdist = k * distance
    factor = inner_product(ns, difference)
    return (
        -1j * k * factor / (4 * distance) * hankel2(1, kdist)
        + 1j * k * z0 * admittance * 1j * hankel2(0, kdist) / 4
    )
