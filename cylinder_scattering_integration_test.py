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
from integrals import admitant_2d_integral
from mesh import Mesh
from pybem import complex_system_matrix, calc_scattered_pressure_at


def calc_coefficiencts(k, radius, rho_c, amplitude, admittance, max_order):
    orders = np.arange(max_order+1)
    thetas = 2*np.pi*orders/len(orders)
    rhs = (amplitude*np.exp(1j*k*radius*np.cos(thetas)) *
           (admittance + np.cos(thetas)/rho_c))
    matrix = np.array([[np.cos(n*theta) * (1j*h2vp(n, k*radius)/rho_c -
                                           admittance*hankel2(n, k*radius))
                        for n in orders]
                       for theta in thetas])
    return np.linalg.solve(matrix, rhs)


def pressure_expansion(k, coefficients, radius, theta):
    return sum(coef * hankel2(n, k*radius) * np.cos(n*theta)
               for coef, n in zip(coefficients, count()))


def radial_velocity_expansion(k, coefficients, rho_c, radius, theta):
    return 1j*sum(coef * h2vp(n, k*radius) * np.cos(n*theta)
                  for coef, n in zip(coefficients, count()))/rho_c


def complex_relative_error(reference, to_test):
    reference, to_test = np.array(reference), np.array(to_test)
    return np.max(np.abs((reference - to_test) / reference))


@pytest.mark.parametrize('ka', [.5, 2])
@pytest.mark.parametrize('admittance', [0, 343])
@pytest.mark.slow
def test_plane_wave_admittance_cylinder_scattering(ka, admittance):
    # set constants
    k = ka  # for radius = 1
    amplitude = 1+1j
    rho, c = 1, 343
    element_size = 1/k/12

    # create mesh
    element_count = int(np.ceil(2*np.pi/element_size))
    element_count = 72 if element_count < 10 else element_count
    angles = np.arange(0, 2*np.pi, 2*np.pi/element_count)
    nodes = [(np.cos(angle), np.sin(angle)) for angle in angles]
    elements = [(i, i+1) for i in range(len(nodes)-1)] + [(len(nodes)-1, 0)]
    mesh = Mesh(nodes, elements, admittance*np.ones(len(elements)))

    # microphone points
    radius = 2
    mic_angles = np.arange(0, 2*np.pi, 2*np.pi/72)
    mic_points = [(radius*np.cos(angle), radius*np.sin(angle))
                  for angle in mic_angles]

    # reference caclculation
    coefficients = calc_coefficiencts(k, 1, rho*c, amplitude, admittance, 80)
    reference_result = [pressure_expansion(k, coefficients, radius, theta)
                        for theta in mic_angles]

    # BEM calculation
    p_incoming = np.array([amplitude*np.exp(1j*k*point[0])
                           for point in mesh.centers])
    matrix = complex_system_matrix(mesh, admitant_2d_integral, k, rho, c)
    surface_pressure = np.linalg.solve(matrix, -p_incoming)
    result = calc_scattered_pressure_at(mesh, admitant_2d_integral, k,
                                        surface_pressure, mic_points, rho, c)

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

    assert complex_relative_error(reference_result, result) < .01
