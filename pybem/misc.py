#!/usr/bin/env python3
"""
2019-10-25 16:00:13
@author: Paul Reiter
"""
import numpy as np


def complex_relative_error(reference, to_test):
    reference, to_test = np.array(reference), np.array(to_test)
    return np.max(np.abs((reference - to_test) / reference))
