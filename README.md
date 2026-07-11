# TEQUILA.jl

TEQUILA is a fixed-boundary equilibrium solver that uses cubic-Hermite finite elements in the radial direction, Fourier modes in the poloidal angle, and a [Miller extended harmonic](https://iopscience.iop.org/article/10.1088/1361-6587/abc63b/meta) (MXH) flux-surface parametrization [[Arbon 2021](https://iopscience.iop.org/article/10.1088/1361-6587/abc63b/meta)]. MXH provides an efficient representation of flux surfaces.

The high-level implementation borrows extensively: Jardin, Stephen. Computational Methods in Plasma Physics. CRC Press, 2010.

The VEQ direct solver (`src/veq.jl` and `src/veq_shot.jl`) is a Julia port of [veqpy](https://github.com/zhangtakeda/veqpy) by rhzhang.

TEQUILA uses COCOS 11 convention.

## Online documentation
For more details, see the [online documentation](https://projecttorreypines.github.io/TEQUILA.jl/dev).

![Docs](https://github.com/ProjectTorreyPines/TEQUILA.jl/actions/workflows/make_docs.yml/badge.svg)