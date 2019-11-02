#!/usr/bin/env python3
"""
2019-05-09 14:28:36
@author: Paul Reiter
"""
import numpy as np
import pytest
from .cylinder_scattering import calc_coefficiencts, pressure_expansion
from pybem import complex_relative_error, Mesh, calc_solution_at
from .integration_test import wrapped_kirchhoff_helmholtz_solver
from pybem.helmholtz import (
    burton_miller_solver,
    admitant_2d_integral,
    fast_burton_miller_solver,
    fast_calc_solution_at,
)


@pytest.mark.parametrize("ka", [0.5, 2])
@pytest.mark.parametrize("admittance", [0, 1 / 343])
@pytest.mark.parametrize(
    "solver, calc_solution",
    [
        (wrapped_kirchhoff_helmholtz_solver, calc_solution_at),
        (burton_miller_solver, calc_solution_at),
        (fast_burton_miller_solver, fast_calc_solution_at),
    ],
)
@pytest.mark.slow
def test_plane_wave_admittance_cylinder_scattering(
    ka, admittance, solver, calc_solution, tmpdir, request
):
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
    surface_pressure = solver(mesh, p_incoming, grad_p_incoming, z0, k)
    result = calc_solution(
        admitant_2d_integral, mesh, surface_pressure, mic_points, z0, k
    )

    # plotting
    if request.config.getoption("--plot"):
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection="polar")

        for result in [result, reference_result]:
            for part in [np.abs, np.real, np.imag]:
                ax.plot(
                    [np.arctan2(point[1], point[0]) for point in mic_points],
                    (part(result)),
                )
        ax.legend(
            [
                "BEM Abs",
                "BEM Re",
                "BEM Im",
                "Expansion Abs",
                "Expansion Re",
                "Expansion Im",
            ]
        )

        fig.savefig(tmpdir / (request.node.name + ".pdf"))
        plt.close(fig)

    assert complex_relative_error(reference_result, result) < 1e-2
