from .helmholtz import (
    g_2d,
    vector_h_2d,
    h_2d,
    hs_2d,
    hypersingular,
    admitant_2d_integral,
)
from .kirchhoff_helmholtz import admitant_2d_matrix_element, kirchhoff_helmholtz_solver
from .burton_miller import (
    admitant_2d_matrix_element_bm,
    burton_miller_rhs,
    burton_miller_solver,
)

# import pyximport

# pyximport.install(language_level=3)
from .fast_helmholtz import fast_calc_solution_at
from .fast_burton_miller import fast_burton_miller_solver
