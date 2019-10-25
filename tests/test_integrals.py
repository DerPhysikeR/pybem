#!/usr/bin/env python3
"""
2019-05-05 12:18:24
@author: Paul Reiter
"""
import numpy as np
from pybem import (
    line_integral,
)


def test_line_integral():

    def square(coords):
        return coords.dot(coords)

    result = line_integral(square, [0, 0], [3, 4], False)
    np.testing.assert_almost_equal(125/3, result)


def test_line_integral_singular():

    def singular_function(coords):
        return 1/coords[0]

    result = line_integral(singular_function, [-1, 0], [1, 0], True)
    np.testing.assert_almost_equal(0, result)
