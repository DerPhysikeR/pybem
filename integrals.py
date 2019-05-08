#!/usr/bin/env python3
"""
2019-05-05 12:18:09
@author: Paul Reiter
"""
import numpy as np
from scipy.integrate import quad, fixed_quad
from scipy.special import hankel2


def line_integral(function, p0, p1, singular):
    x0, y0, x1, y1 = *p0, *p1
    length = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)

    def to_quad(t):
        return function(np.array([x0 + t*(x1-x0), y0 + t*(y1-y0)]))

    if singular:
        return length*complex_quad(to_quad, 0, 1, points=[.5])
    else:
        return length*fixed_quad(np.vectorize(to_quad), 0., 1.)[0]


def complex_quad(function, *args, **kwargs):

    def real_function(*real_args, **real_kwargs):
        return np.real(function(*real_args, **real_kwargs))

    def imag_function(*imag_args, **imag_kwargs):
        return np.imag(function(*imag_args, **imag_kwargs))

    real_part = quad(real_function, *args, **kwargs)[0]
    imag_part = quad(imag_function, *args, **kwargs)[0]

    return real_part + 1j*imag_part


def admitant_2d_integral(k, r, admittance, n, corners, singular, rho, c):
    def integral_function(rs):
        return h_2d(n, k, r, rs) + 1j*k*c*rho*admittance*g_2d(k, r, rs)
    return (line_integral(integral_function, corners[0], corners[1], singular)
            - singular/2)


def h_2d(n, k, r, rs):
    distance = np.sqrt((r-rs).dot(r-rs))
    scaling = 1j*k*n.dot(r - rs)/distance/8
    return scaling*(hankel2(-1, k*distance) - hankel2(1, k*distance))


def g_2d(k, r, rs):
    return 1j*hankel2(0, k*np.sqrt((r-rs).dot(r-rs)))/4
