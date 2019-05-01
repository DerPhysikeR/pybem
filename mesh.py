#!/usr/bin/env python3
"""
2019-05-01 09:37:10
@author: Paul Reiter
"""
import numpy as np


class Mesh:

    def __init__(self, nodes, elements):
        self.nodes = np.array(nodes)
        self.elements = np.array(elements)
        self._centers = None
        self._normals = None

    @property
    def centers(self):
        if self._centers is None:
            self._centers = np.empty((self.elements.shape[0],
                                      self.nodes.shape[1]))
            for i, corners in enumerate(self.elements):
                self._centers[i] = np.vstack([self.nodes[j]
                                              for j in corners]).mean(axis=0)
        return self._centers

    @property
    def normals(self):
        if self._normals is None:
            if self.nodes.shape[1] == 2:
                self._normals = self.calc_2d_normals()
            elif self.nodes.shape[1] == 3:
                self._normals = self.calc_3d_normals()
            else:
                raise ValueError('Mesh is neither 2D nor 3D!')
        return self._normals

    def calc_2d_normals(self):
        normals = np.empty((self.elements.shape[0], self.nodes.shape[1]))
        for i, corners in enumerate(self.elements):
            element_vector = self.nodes[corners[1]] - self.nodes[corners[0]]
            element_length = np.sqrt(element_vector.dot(element_vector))
            normals[i] = element_vector.dot(
                np.array([[0, -1], [1, 0]]))/element_length
        return normals

    def calc_3d_normals(self):
        raise NotImplementedError('3D normals are not supported yet!')
