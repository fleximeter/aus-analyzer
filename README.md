# aus-analyzer

This crate provides Python bindings for some functionality in the `aus` crate. The purpose is to perform spectral analysis much more quickly.

To build the crate, you need to have `maturin` installed as a Python package in your virtual environment. If you run `maturin develop`, it will build this crate as a Python package and install it in the virtual environment. If you run `maturin build`, it will build this crate to a wheel that you can install elsewhere.

At present, it appears that the function header comments in `aus_analyzer.pyi` do not carry through to Visual Studio Code. You can consult that file for descriptions of the functions and their parameters. The parameter names and type annotations do carry over, though.

## License
This crate is dual-licensed under the MIT and GPL 3.0 (or any later version) licenses. You can choose between one of them if you use this crate.

`SPDX-License-Identifier: MIT OR GPL-3.0-or-later`
