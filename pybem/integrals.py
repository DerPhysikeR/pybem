#!/usr/bin/env python3
"""
2019-05-05 12:18:09
@author: Paul Reiter
"""
import numpy as np
from scipy.integrate import quad, fixed_quad


def line_integral(function, p0, p1, singular):
    line_vector = p1 - p0
    length = np.sqrt(line_vector.dot(line_vector))

    def to_quad(t):
        return function((p1 + p0) / 2 + t * (p1 - p0) / 2)

    if singular:
        return length * complex_quad(to_quad, -1, 1, points=[0]) / 2
    return length * fixed_quad(np.vectorize(to_quad), -1.0, 1.0)[0] / 2


def complex_quad(function, *args, **kwargs):
    def real_function(*real_args, **real_kwargs):
        return np.real(function(*real_args, **real_kwargs))

    def imag_function(*imag_args, **imag_kwargs):
        return np.imag(function(*imag_args, **imag_kwargs))

    real_part = quad(real_function, *args, **kwargs)[0]
    imag_part = quad(imag_function, *args, **kwargs)[0]

    return real_part + 1j * imag_part
