#!/usr/bin/env python3
"""
2019-05-01 09:36:55
@author: Paul Reiter
"""
import numpy as np


def complex_system_matrix(mesh, integral_function, k, *args, **kwargs):
    dimension = len(mesh.elements)
    system_matrix = np.empty((dimension, dimension), dtype=complex)
    for i, row in enumerate(system_matrix):
        for j, col in enumerate(row):
            system_matrix[i, j] = \
                integral_function(k, mesh.centers[i], mesh.admittances[j],
                                  mesh.normals[j], mesh.corners[j], i == j,
                                  *args, **kwargs)
    return system_matrix
