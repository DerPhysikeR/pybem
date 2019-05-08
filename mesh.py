#!/usr/bin/env python3
"""
2019-05-01 09:37:10
@author: Paul Reiter
"""
import numpy as np


class Mesh:

    def __init__(self, nodes, elements, admittances=None):
        self._nodes = np.array(nodes, dtype=float)
        assert self._nodes.shape[1] == 2
        self._elements = np.array(elements, dtype=int)
        assert self._elements.shape[1] == 2
        self._corners = self._calc_corners()
        self._centers = self._calc_centers()
        self._normals = self._calc_normals()
        if admittances is None:
            self._admittances = np.zeros(len(elements), dtype=complex)
        else:
            assert len(admittances) == len(elements)
            self._admittances = np.array(admittances, dtype=complex)

    @property
    def nodes(self):
        return self._nodes

    @property
    def elements(self):
        return self._elements

    @property
    def centers(self):
        return self._centers

    @property
    def normals(self):
        return self._normals

    @property
    def corners(self):
        return self._corners

    @property
    def admittances(self):
        return self._admittances

    def _calc_corners(self):
        corners = np.empty((*self._elements.shape, self._nodes.shape[1]),
                           dtype=float)
        for i, corner_indexes in enumerate(self._elements):
            corners[i] = np.vstack([self._nodes[j] for j in corner_indexes])
        return corners

    def _calc_centers(self):
        return self._corners.mean(axis=1)

    def _calc_normals(self):
        normals = np.empty((self._elements.shape[0], self._nodes.shape[1]),
                           dtype=float)
        for i, corners in enumerate(self._corners):
            element_vector = corners[1] - corners[0]
            element_length = np.sqrt(element_vector.dot(element_vector))
            normals[i] = element_vector.dot(
                np.array([[0., -1.], [1., 0.]]))/element_length
        return normals
