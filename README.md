# `MaCh3 SBI Tools` Simulation Based Inference with Neutrinos
[![License](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![codecov](https://codecov.io/github/henry-wallace-phys/mach3sbitools/graph/badge.svg?token=LY1UV4USFH)](https://codecov.io/github/henry-wallace-phys/mach3sbitools)
[![Code - Documented](https://img.shields.io/badge/Code-Documented-2ea44f)](https://henry-wallace-phys.github.io/MaCh3SbiTools)
[![unit-test](https://github.com/henry-wallace-phys/MaCh3SbiTools/actions/workflows/pytest.yml/badge.svg)](https://github.com/henry-wallace-phys/MaCh3SbiTools/actions/workflows/pytest.yml)
[![CodeQL](https://github.com/henry-wallace-phys/MaCh3SbiTools/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/henry-wallace-phys/MaCh3SbiTools/actions/workflows/github-code-scanning/codeql)
[![mypy-typecheck](https://github.com/henry-wallace-phys/MaCh3SbiTools/actions/workflows/mypy.yml/badge.svg)](https://github.com/henry-wallace-phys/MaCh3SbiTools/actions/workflows/mypy.yml)
[![ruff-lint](https://github.com/henry-wallace-phys/MaCh3SbiTools/actions/workflows/ruff.yml/badge.svg)](https://github.com/henry-wallace-phys/MaCh3SbiTools/actions/workflows/ruff.yml)
[![Build & Deploy Sphinx Docs](https://github.com/henry-wallace-phys/MaCh3SbiTools/actions/workflows/docs.yml/badge.svg)](https://github.com/henry-wallace-phys/MaCh3SbiTools/actions/workflows/docs.yml)

MaCh3 SBI Tools is a package used to perform
Bayesian Simulation based inference with  a flexible simulator and training setup using tools from the [SBI](https://github.com/sbi-dev/sbi) [[1](#References)] package. The simulator is designed to work primarily with [MaCh3](https://github.com/mach3-software/MaCh3/tree/develop) [[2](#References)].

For full documentation see: https://henry-wallace-phys.github.io/MaCh3SbiTools/

## Install
`mach3sbitools` requires python `3.11` or higher. It can be compiled for usage on a GPU which requires the appropriate [pyTorch install](https://pytorch.org/get-started/locally/). It is recommended to either use a `virtual environement` or `Conda`. 

**NOTE**: Whilst this package can be installed with `uv` external packages such as [`MaCh3`](https://github.com/mach3-software/MaCh3/tree/develop) [[2](#References)] may not work within that framework

### With PIP
```sh
python -m pip install [-e] .
```

### With Conda
```shell
conda develop .
```

## Tutorials
to be added

## References
[1] Boelts, J. et al. (2025). sbi reloaded: a toolkit for simulation-based inference workflows. Journal of Open Source Software, 10(108), 7754. https://doi.org/10.21105/joss.07754

[2] The MaCh3 Collaboration. (2026). mach3-software/MaCh3: v2.4.1 (v2.4.1). Zenodo. https://doi.org/10.5281/zenodo.18627288
