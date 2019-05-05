#!/usr/bin/env python3
"""
2019-05-01 09:48:40
@author: Paul Reiter
"""
import numpy as np
import pytest
from mesh import Mesh


@pytest.mark.parametrize('nodes, elements, center', [
    ([[0, 0], [1, 0], [1, 1]], [[0, 1], [1, 2]], np.array([[.5, 0], [1, .5]])),
])
def test_mesh_centers(nodes, elements, center):
    np.testing.assert_allclose(center, Mesh(nodes, elements).centers)


@pytest.mark.parametrize('nodes, elements, normals', [
    ([[0, 0], [2, 0], [2, 2]], [[0, 1], [1, 2]], np.array([[0, -1], [1, 0]])),
])
def test_mesh_normals(nodes, elements, normals):
    np.testing.assert_allclose(normals, Mesh(nodes, elements).normals)
