# cse355-machine-design

An open-source, high-performance library for defining, simulating, visualizing, and interacting with automata and Turing machines in Arizona State University's CSE 355: Introduction to Theoretical Computer Science courses.
It is implemented in Rust and is available as a [Rust crate](https://crates.io/crates/cse355-machine-design) and [Python package](https://pypi.org/project/cse355-machine-design).


## Getting Started

If you want to use the Python library (e.g., if you are a student in ASU's CSE 355), refer to the documentation on [PyPI](https://pypi.org/project/cse355-machine-design).
If you want to use the Rust crate in another Rust project, refer to the [docs.rs](https://docs.rs/cse355-machine-design) documentation for installation and usage examples.

Otherwise, clone/download this repository if you want to:

- Build and run tests and benchmarks
- Build the Python package locally
- Contribute/develop new features or bug fixes

> [!WARNING]
> Again, if you are an ASU student trying to use the repository for assignments in CSE 355, you do not need to clone or download this repository.
> You just need to install the Python package; instructions for this are found on [PyPi](https://pypi.org/project/cse355-machine-design).


### Requirements

This project supports Linux, macOS, and Windows.
You need Rust, installed either [using `rustup`](https://www.rust-lang.org/tools/install) or via your system package manager of choice.
This provides the `cargo` build system and dependency manager for compilation, testing, benchmarking, documentation, and packaging.


### Tests and Benchmarks

`cse355-machine-design` comes with a variety of unit, integration, and documentation example tests ensuring the correct functionality.
To run all tests, use:

```shell
cargo test
```

To measure library performance, we've implemented benchmarks using the [`criterion`](https://crates.io/crates/criterion) crate.
To run all benchmarks, use:

```shell
cargo bench
```

See the [`criterion` command line options](https://bheisler.github.io/criterion.rs/book/user_guide/command_line_options.html) for details on how to run only specific benchmarks or save baselines for comparison.


### Building the Python Package Locally

We use [`pyo3`](https://crates.io/crates/pyo3) to package functionality from our Rust crate as a Python package called `cse355_machine_design`.
To build this package locally, first create a virtual environment for this project using a manager of your choice.
Then install [`maturin`](https://pypi.org/project/maturin/):

```shell
pip install maturin      # using pip
pipx install maturin     # using pipx
uv tool install maturin  # using uv
```

Within the virtual environment, build and install this project as a Python package:

```shell
maturin develop --release
```

> [!TODO]
> Add a Python usage example.

See the [`cse355_machine_design::python` documentation](https://docs.rs/cse355-machine-design/latest/cse355_machine_design/python) for a complete list of functions exposed to the Python package along with usage examples.

To run the Python test suite, install [`pytest`](https://pypi.org/project/pytest/) in your virtual environment and then simply run `pytest`.


## Contributing

Have a suggestion for new features or a bug you need fixed?
Open a [new issue](https://github.com/DaymudeLab/assembly-theory/issues/new).

Want to contribute your own code?

- Familiarize yourself with the [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/checklist.html) and overall architecture of `cse355-machine-design`.
- Development team members should work in individual feature branches.
External contributors should work in repository forks.
- Commit messages should follow [conventional commits](https://www.conventionalcommits.org).
- Before opening a pull request onto `main`, make sure you rebase onto `main`, run `cargo fmt`, and resolve any issues raised by `cargo clippy`.
- Open a [new pull request](https://github.com/DaymudeLab/cse355-machine-design/compare), provide a descriptive list of your changes (with references to any issues your PR resolves), and assign [@jdaymude](https://github.com/jdaymude) as a reviewer. 
Your PR will not be reviewed unless it passes all GitHub Actions (compilation, formatting, tests, etc.).


## Governance

`cse355-machine-design` was originally developed by Saajan Maslanka ([@SaajanM](https://github.com/SaajanM)) under the supervision of Joshua J. Daymude ([@jdaymude](https://github.com/jdaymude)) and is now maintained solely by Joshua J. Daymude.


## License

`cse355-machine-design` is licensed under the [Apache License, Version 2.0](https://choosealicense.com/licenses/apache-2.0/) or the [MIT License](https://choosealicense.com/licenses/mit/), at your option.

Unless you explicitly state otherwise, any contribution you intentionally submit for inclusion in this repository (as defined by Apache-2.0) shall be dual-licensed as above, without any additional terms or conditions.
