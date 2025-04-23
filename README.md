# SeqOpt.jl

[![Build Status](https://github.com/blegat/SeqOpt.jl/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/blegat/SeqOpt.jl/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/blegat/SeqOpt.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/blegat/SeqOpt.jl)

**WARNING**: This package is still at its early stage of development.

[SeqOpt.jl](https://github.com/blegat/SeqOpt.jl) is a sequential solver for nonlinear optimization.
It implements a generalization of [Sequential quadratic programming](https://en.wikipedia.org/wiki/Sequential_quadratic_programming) to nonlinear conic programs developed in:

> Torbjørn Cunis & Benoît Legat. Sequential sum-of-squares programming for analysis of nonlinear systems. Submitted to the 2023 American Control Conference, arXiv:2210.02142.

## License

`SeqOpt.jl` is licensed under the [MIT License](https://github.com/blegat/SeqOpt.jl/blob/main/LICENSE.md).

## Installation

SeqOpt currently relies on the in-development PR https://github.com/jump-dev/MathOptInterface.jl/pull/2059.
Install SeqOpt using `Pkg.add`:

```julia
import Pkg
Pkg.add(PackageSpec(name="MathOptInterface", rev="od/nlp-expr"))
Pkg.add("https://github.com/blegat/SeqOpt.jl")
```

## Use with JuMP

Use `SeqOpt` with JuMP as follows:

```julia
using JuMP
import SCS
import SeqOpt
model = JuMP.Model(() -> SeqOpt.Optimizer(SCS.Optimizer))
```

Replace `SCS.Optimizer` with an optimizer capable of solving a
linearized version of your optimization problem.

## Citing

See [CITATION.bib](https://github.com/blegat/SeqOpt.jl/blob/main/CITATION.bib).
