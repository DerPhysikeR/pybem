#!/usr/bin/env python3
"""
2019-05-01 09:37:10
@author: Paul Reiter
"""
import numpy as np


class Mesh:

    def __init__(self, nodes, elements):
        self._nodes = np.array(nodes)
        assert self._nodes.shape[1] == 2
        self._elements = np.array(elements)
        assert self._elements.shape[1] == 2
        self._centers = self._calc_centers()
        self._normals = self._calc_normals()

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

    def _calc_centers(self):
        centers = np.empty((self._elements.shape[0], self._nodes.shape[1]))
        for i, corners in enumerate(self._elements):
            centers[i] = np.vstack([self._nodes[j]
                                    for j in corners]).mean(axis=0)
        return centers

    def _calc_normals(self):
        normals = np.empty((self._elements.shape[0], self._nodes.shape[1]))
        for i, corners in enumerate(self._elements):
            element_vector = self._nodes[corners[1]] - self._nodes[corners[0]]
            element_length = np.sqrt(element_vector.dot(element_vector))
            normals[i] = element_vector.dot(
                np.array([[0, -1], [1, 0]]))/element_length
        return normals
