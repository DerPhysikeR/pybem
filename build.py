from Cython.Build import cythonize

modules = [
    "pybem/helmholtz/fast_burton_miller.pyx",
    "pybem/helmholtz/fast_helmholtz.pyx",
]

extensions = cythonize(modules, language_level="3")


def build(setup_kwargs):
    """Needed for the poetry building interface."""

    setup_kwargs.update({"ext_modules": extensions})
