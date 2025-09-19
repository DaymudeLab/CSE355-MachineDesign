//! Expose public `cse355_machine_design` functionality to a Python package
//! using [`pyo3`](https://docs.rs/pyo3/latest/pyo3/).
//!
//! The package is available [on PyPI](https://pypi.org/project/cse355-machine-design);
//! see that README for installation and usage instructions. To build the
//! Python package directly from this crate's source code, see the instructions
//! in the [GitHub README](https://github.com/DaymudeLab/cse355-machine-design).
//!
//! Note that all Rust functions in this module have the form `_fn_name`, which
//! correspond to the actual Rust function `fn_name` elsewhere in the crate and
//! are exposed to the Python package as `fn_name`.
//!
//! # Python Example
//!
//! ```custom,{class=language-python}
//! TODO
//! ```
