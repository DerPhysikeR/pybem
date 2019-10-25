#!/usr/bin/env python3
"""
2019-05-05 12:18:09
@author: Paul Reiter
"""
import numpy as np
from scipy.integrate import quad, fixed_quad
from scipy.special import hankel2, struve


def line_integral(function, p0, p1, singular):
    x0, y0, x1, y1 = *p0, *p1
    length = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)

    def to_quad(t):
        return function(np.array([(x1+x0)/2 + t*(x1-x0)/2,
                                  (y1+y0)/2 + t*(y1-y0)/2]))

    if singular:
        return length*complex_quad(to_quad, -1, 1, points=[0])/2
    return length*fixed_quad(np.vectorize(to_quad), -1., 1.)[0]/2


def complex_quad(function, *args, **kwargs):

    def real_function(*real_args, **real_kwargs):
        return np.real(function(*real_args, **real_kwargs))

    def imag_function(*imag_args, **imag_kwargs):
        return np.imag(function(*imag_args, **imag_kwargs))

    real_part = quad(real_function, *args, **kwargs)[0]
    imag_part = quad(imag_function, *args, **kwargs)[0]

    return real_part + 1j*imag_part


def admitant_2d_integral(k, point, mesh, idx, rho, c):
    n = mesh.normals[idx]
    admittance = mesh.admittances[idx]
    corners = mesh.corners[idx]

    def integral_function(rs):
        return (- hs_2d(n, k, point, rs)
                + 1j*k*c*rho*admittance*g_2d(k, point, rs))

    return (line_integral(integral_function, corners[0], corners[1], False))


def admitant_2d_matrix_element(k, mesh, row_idx, col_idx, rho, c):
    ns, r = mesh.normals[col_idx], mesh.centers[row_idx]
    corners, admittance = mesh.corners[col_idx], mesh.admittances[col_idx]
    singular = row_idx == col_idx

    def integral_function(rs):
        return hs_2d(ns, k, r, rs) - 1j*k*c*rho*admittance*g_2d(k, r, rs)

    return (line_integral(integral_function, corners[0], corners[1], singular)
            + singular/2)


def regularized_hypersingular_bm_part(v):
    return (
        hankel2(1, v)/v
        - 2j/(np.pi*v**2)  # regularization
    )


def admitant_2d_matrix_element_bm(k, mesh, row_idx, col_idx, rho, c,
                                  coupling_sign):
    n, ns = mesh.normals[row_idx], mesh.normals[col_idx]
    r = mesh.centers[row_idx]
    corners, admittance = mesh.corners[col_idx], mesh.admittances[col_idx]
    singular = row_idx == col_idx
    z0 = rho*c

    if singular:
        element_vector = corners[1] - corners[0]
        length = np.sqrt(element_vector.dot(element_vector))
        lk2 = length*k/2
        return (
            + z0*admittance*np.pi*length*k/8 * (
                + hankel2(0, lk2)*struve(-1, lk2)
                + hankel2(1, lk2)*struve(0, lk2)
            )
            - coupling_sign*fixed_quad(regularized_hypersingular_bm_part, 0, lk2)[0]/2
            + coupling_sign*2j/(np.pi*k*length)
            + singular*(1 - coupling_sign*z0*admittance)/2
        )

    else:
        def integral_function(rs):
            return (
                hs_2d(ns, k, r, rs)
                - 1j*k*z0*admittance*g_2d(k, r, rs)
                + coupling_sign*z0*admittance*h_2d(n, k, r, rs)
                + coupling_sign*1j/k * hypersingular(k, r, rs, n, ns)
            )

        return (line_integral(integral_function, corners[0], corners[1], singular)
                + singular*(1 - coupling_sign*z0*admittance)/2)


def hypersingular(k, r, rs, n, ns):
    vector = r-rs
    distance = np.sqrt(vector.dot(vector))
    kdist = k*distance
    return 1j*k/(4*distance**2) * (
        distance*hankel2(1, kdist)*n.dot(ns)
        - k*hankel2(2, kdist)*n.dot(vector)*ns.dot(vector)
    )


def burton_miller_rhs(k, mesh, p_inc, grad_p_inc, coupling_sign=-1):
    return (
        p_inc
        + (coupling_sign*1j/k)*(grad_p_inc * mesh.normals).sum(axis=1)
    )


def vector_h_2d(k, r, rs):
    """Vectorial gradient of the 2D Green's function acoording to the obverver
       point r"""
    distance = np.sqrt((r-rs).dot(r-rs))
    return -1j*k*(r-rs)/(4*distance) * hankel2(1, k*distance)


def hs_2d(ns, k, r, rs):
    """Gradient of the 2D Green's function according to the source point rs"""
    return -h_2d(ns, k, r, rs)


def h_2d(n, k, r, rs):
    """Gradient of the 2D Green's function according to the obverver point r"""
    return n.dot(vector_h_2d(k, r, rs))


def g_2d(k, r, rs):
    """2D Green's function"""
    return 1j*hankel2(0, k*np.sqrt((r-rs).dot(r-rs)))/4
