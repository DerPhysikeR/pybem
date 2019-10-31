from .kirchhoff_helmholtz import (
    g_2d,
    hs_2d,
    vector_h_2d,
    admitant_2d_integral,
    admitant_2d_matrix_element,
    kirchhoff_helmholtz_solver,
)
from .burton_miller import (
    h_2d,
    hypersingular,
    admitant_2d_matrix_element_bm,
    burton_miller_rhs,
    burton_miller_solver,
)

import pyximport

pyximport.install(language_level=3)
from .fast_burton_miller import fast_burton_miller_solver
