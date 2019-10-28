#!/usr/bin/env python3
"""
2019-05-01 09:36:55
@author: Paul Reiter
"""
import numpy as np


def complex_system_matrix(matrix_element_function, mesh, *args, **kwargs):
    dimension = len(mesh.elements)
    system_matrix = np.empty((dimension, dimension), dtype=complex)
    for i, row in enumerate(system_matrix):
        for j, col in enumerate(row):
            system_matrix[i, j] = matrix_element_function(mesh, i, j, *args, **kwargs)
    return system_matrix


def calc_solution_at(
    integral_function, mesh, surface_solution, points_of_interest, *args, **kwargs
):
    assert len(mesh.elements) == len(surface_solution)
    points_of_interest = np.array(points_of_interest, dtype=float)
    solution = np.zeros(len(points_of_interest), dtype=complex)
    for i, point in enumerate(points_of_interest):
        for j, sp in enumerate(surface_solution):
            solution[i] += sp * integral_function(mesh, j, point, *args, **kwargs)
    return solution
