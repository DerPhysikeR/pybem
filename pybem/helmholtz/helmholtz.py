#!/usr/bin/env python3
"""
2019-11-01 09:54:09
@author: Paul Reiter
"""
import numpy as np
from scipy.special import hankel2
from ..integrals import line_integral


def g_2d(k, r, rs):
    """2D Green's function"""
    return 1j * hankel2(0, k * np.sqrt((r - rs).dot(r - rs))) / 4


def vector_h_2d(k, r, rs):
    """Vectorial gradient of the 2D Green's function acoording to the obverver
       point r"""
    distance = np.sqrt((r - rs).dot(r - rs))
    return -1j * k * (r - rs) / (4 * distance) * hankel2(1, k * distance)


def h_2d(n, k, r, rs):
    """Gradient of the 2D Green's function according to the obverver point r"""
    return -hs_2d(n, k, r, rs)


def hs_2d(ns, k, r, rs):
    """Gradient of the 2D Green's function according to the source point rs"""
    return -ns.dot(vector_h_2d(k, r, rs))


def hypersingular(k, r, rs, n, ns):
    vector = r - rs
    distance = np.sqrt(vector.dot(vector))
    kdist = k * distance
    return (1j * k / (4 * distance ** 2)) * (
        distance * hankel2(1, kdist) * n.dot(ns)
        - k * hankel2(2, kdist) * n.dot(vector) * ns.dot(vector)
    )


def admitant_2d_integral(mesh, idx, point, z0, k):
    n = mesh.normals[idx]
    admittance = mesh.admittances[idx]
    corners = mesh.corners[idx]

    def integral_function(rs):
        return -hs_2d(n, k, point, rs) + 1j * k * z0 * admittance * g_2d(k, point, rs)

    return line_integral(integral_function, corners[0], corners[1], False)
