#!/usr/bin/env python3
"""
2019-05-09 14:28:36
@author: Paul Reiter
"""
from itertools import count
import numpy as np
from scipy.special import hankel2, h2vp
import pytest

# import matplotlib.pyplot as plt
from pybem import complex_relative_error, Mesh, calc_solution_at
from pybem.helmholtz import (
    kirchhoff_helmholtz_solver,
    burton_miller_solver,
    admitant_2d_integral,
)


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


@pytest.mark.parametrize("ka", [0.5, 2])
@pytest.mark.parametrize("admittance", [0, 1 / 343])
@pytest.mark.parametrize("solver", [kirchhoff_helmholtz_solver, burton_miller_solver])
@pytest.mark.slow
def test_plane_wave_admittance_cylinder_scattering(ka, admittance, solver):
    # set constants
    k = ka  # for radius = 1
    amplitude = 1 + 1j
    z0 = 343
    element_size = 1 / k / 16

    # create mesh
    element_count = int(np.ceil(2 * np.pi / element_size))
    min_count = 180
    element_count = min_count if element_count < min_count else element_count
    angles = np.arange(0, 2 * np.pi, 2 * np.pi / element_count)
    nodes = [(np.cos(angle), np.sin(angle)) for angle in angles]
    elements = [(i, i + 1) for i in range(len(nodes) - 1)] + [(len(nodes) - 1, 0)]
    mesh = Mesh(nodes, elements, admittance * np.ones(len(elements)))

    # microphone points
    radius = 2
    mic_angles = np.arange(0, 2 * np.pi, 2 * np.pi / 72)
    mic_points = [
        (radius * np.cos(angle), radius * np.sin(angle)) for angle in mic_angles
    ]

    # reference caclculation
    coefficients = calc_coefficiencts(k, 1, z0, amplitude, admittance, 100)
    reference_result = [
        pressure_expansion(k, coefficients, radius, theta) for theta in mic_angles
    ]

    # BEM calculation
    p_incoming = np.array(
        [amplitude * np.exp(1j * k * point[0]) for point in mesh.centers]
    )
    grad_p_incoming = np.array(
        [[1j * k * amplitude * np.exp(1j * k * point[0]), 0] for point in mesh.centers]
    )
    surface_pressure = solver(mesh, p_incoming, grad_p_incoming, k, z0)
    result = calc_solution_at(
        admitant_2d_integral, mesh, surface_pressure, mic_points, k, z0
    )

    # # plotting
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1, projection='polar')

    # # real and imaginary parts of the solution
    # ax.plot([np.arctan2(point[1], point[0]) for point in mic_points],
    #         (np.real(result)))
    # ax.plot([np.arctan2(point[1], point[0]) for point in mic_points],
    #         (np.imag(result)))
    # ax.plot([np.arctan2(point[1], point[0]) for point in mic_points],
    #         (np.real(reference_result)))
    # ax.plot([np.arctan2(point[1], point[0]) for point in mic_points],
    #         (np.imag(reference_result)))
    # ax.legend(['BEM Re', 'BEM Im', 'Expansion Re', 'Expansion Im'])

    # # # absolute value of solution
    # # ax.plot([np.arctan2(point[1], point[0]) for point in mic_points],
    # #         (np.abs(result)))
    # # ax.plot([np.arctan2(point[1], point[0]) for point in mic_points],
    # #         (np.abs(reference_result)))
    # # ax.legend(['BEM Abs', 'Expansion Abs'])

    # fig.savefig('surface_pressure_distribution.pdf')
    # plt.close(fig)

    assert complex_relative_error(reference_result, result) < 1e-2
