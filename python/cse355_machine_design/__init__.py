from .cse355_machine_design import *

# Currently, maturin and pyo3 are working together to directly expose functions
# in src/python.rs marked with #[pyfunction] to this cse355_machine_design
# Python. If ever we need Python-specific wrapper code, we can write those
# extensions here.

# Make cse355_machine_design docstrings and functions accessible using:
# '>>> import cse355_machine_design'
# instead of requiring:
# '>>> from cse355_machine_design import cse355_machine_design'
__doc__ = cse355_machine_design.__doc__
if hasattr(cse355_machine_design, "__all__"):
    __all__ = cse355_machine_design.__all__
