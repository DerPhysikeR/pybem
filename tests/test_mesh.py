#!/usr/bin/env python3
"""
2019-05-01 09:48:40
@author: Paul Reiter
"""
import numpy as np
import pytest
from pybem.mesh import (
    Mesh,
)


@pytest.mark.parametrize('nodes, elements, corners', [
    ([[0, 0], [1, 0], [1, 1]], [[0, 1], [1, 2]],
     np.array([[[0, 0], [1, 0]], [[1, 0], [1, 1]]])),
])
def test_mesh_corners(nodes, elements, corners):
    np.testing.assert_allclose(corners, Mesh(nodes, elements).corners)


@pytest.mark.parametrize('nodes, elements, center', [
    ([[0, 0], [1, 0], [1, 4]], [[0, 1], [1, 2]], np.array([[.5, 0], [1, 2]])),
    ([[0, 1], [2, 3]], [[0, 1]], np.array([[1, 2]])),
])
def test_mesh_centers(nodes, elements, center):
    np.testing.assert_allclose(center, Mesh(nodes, elements).centers)


@pytest.mark.parametrize('nodes, elements, normals', [
    ([[0, 0], [2, 0], [2, 2]], [[0, 1], [1, 2]], np.array([[0, -1], [1, 0]])),
])
def test_mesh_normals(nodes, elements, normals):
    np.testing.assert_allclose(normals, Mesh(nodes, elements).normals)


@pytest.mark.parametrize('nodes, elements, admittances, reference', [
    ([[0, 0], [1, 0]], [[0, 1]], [1], np.array([1], dtype=complex)),
    ([[0, 0], [1, 0]], [[0, 1]], None, np.array([0], dtype=complex)),
])
def test_mesh_admittances(nodes, elements, admittances, reference):
    np.testing.assert_allclose(reference, Mesh(nodes, elements,
                                               admittances).admittances)


@pytest.mark.parametrize('nodes, elements, admittances, reference', [
    ([[0, 0], [1, 0]], [[0, 1]], [1], np.array([1], dtype=complex)),
    ([[0, 0], [1, 0]], [[0, 1]], None, np.array([0], dtype=complex)),
])
def test_mesh_set_admittances(nodes, elements, admittances, reference):
    mesh = Mesh(nodes, elements)
    mesh.admittances = admittances
    np.testing.assert_allclose(reference, mesh.admittances)
