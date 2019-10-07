#!/usr/bin/env python3
"""
2019-05-01 09:36:55
@author: Paul Reiter
"""
import numpy as np
import pybem as pb


def complex_system_matrix(mesh, matrix_element_function, k, *args, **kwargs):
    dimension = len(mesh.elements)
    system_matrix = np.empty((dimension, dimension), dtype=complex)
    for i, row in enumerate(system_matrix):
        for j, col in enumerate(row):
            system_matrix[i, j] = \
                matrix_element_function(k, mesh, i, j, *args, **kwargs)
    return system_matrix


def calc_scattered_pressure_at(mesh, integral_function, k, surface_pressure,
                               microphone_points, *args, **kwargs):
    assert len(mesh.elements) == len(surface_pressure)
    microphone_points = np.array(microphone_points, dtype=float)
    solution = np.zeros(len(microphone_points), dtype=complex)
    for i, point in enumerate(microphone_points):
        for j, sp in enumerate(surface_pressure):
            solution[i] += sp*integral_function(k, point, mesh, j,
                                                *args, **kwargs)
    return solution


def simple_solver(mesh, p_incoming, grad_p_incoming, k, rho, c):
    matrix = pb.complex_system_matrix(mesh, pb.admitant_2d_matrix_element, k,
                                      rho, c)
    return np.linalg.solve(matrix, p_incoming)


def burton_miller_solver(mesh, p_incoming, grad_p_incoming, k, rho, c):
    matrix = pb.complex_system_matrix(mesh, pb.admitant_2d_matrix_element_bm,
                                      k, rho, c)
    rhs = pb.burton_miller_rhs(k, mesh, p_incoming, grad_p_incoming)
    return np.linalg.solve(matrix, rhs)
