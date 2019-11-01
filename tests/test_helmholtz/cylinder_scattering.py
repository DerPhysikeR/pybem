#!/usr/bin/env python3
"""
2019-11-01 11:27:25
@author: Paul Reiter
"""
from itertools import count
from scipy.special import hankel2, h2vp
import numpy as np


def calc_coefficiencts(k, radius, z0, amplitude, admittance, max_order):
    orders = np.arange(max_order + 1)
    thetas = np.pi * orders / len(orders)
    rhs = (
        amplitude
        * np.exp(1j * k * radius * np.cos(thetas))
        * (np.cos(thetas) / z0 - admittance)
    )
    matrix = np.array(
        [
            [
                np.cos(n * theta)
                * (1j * h2vp(n, k * radius) / z0 + admittance * hankel2(n, k * radius))
                for n in orders
            ]
            for theta in thetas
        ]
    )
    return np.linalg.solve(matrix, rhs)


def pressure_expansion(k, coefficients, radius, theta):
    return sum(
        coef * hankel2(n, k * radius) * np.cos(n * theta)
        for coef, n in zip(coefficients, count())
    )


def radial_velocity_expansion(k, coefficients, z0, radius, theta):
    return (
        1j
        * sum(
            coef * h2vp(n, k * radius) * np.cos(n * theta)
            for coef, n in zip(coefficients, count())
        )
        / z0
    )
