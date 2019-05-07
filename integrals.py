#!/usr/bin/env python3
"""
2019-05-05 12:18:09
@author: Paul Reiter
"""
import numpy as np
from scipy.integrate import quad
from scipy.special import hankel2


def line_integral(function, p0, p1, singular):
    x0, y0, x1, y1 = *p0, *p1
    length = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)

    def to_quad(t):
        return function(np.array([x0 + t*(x1-x0), y0 + t*(y1-y0)]))

    points = [.5] if singular else None
    return length*quad(to_quad, 0, 1, points=points)[0]


def h_2d(n, k, r, rs):
    distance = np.sqrt((r-rs).dot(r-rs))
    scaling = 1j*k*n.dot(rs - r)/distance/8
    return scaling*(hankel2(-1, k*distance) - hankel2(1, k*distance))


def g_2d(k, r, rs):
    return 1j*hankel2(0, k*np.sqrt((r-rs).dot(r-rs)))/4
