#!/usr/bin/env python3
"""
2019-05-05 12:18:09
@author: Paul Reiter
"""
import numpy as np
from scipy.special import hankel2


def h_2d(n, k, r, rs):
    distance = np.sqrt((r-rs).dot(r-rs))
    scaling = 1j*k*n.dot(rs - r)/distance/8
    return scaling*(hankel2(-1, k*distance) - hankel2(1, k*distance))


def g_2d(k, r, rs):
    return 1j*hankel2(0, k*np.sqrt((r-rs).dot(r-rs)))/4
