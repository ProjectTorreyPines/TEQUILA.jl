__precompile__()
module TEQUILA

using FiniteElementHermite
using MillerExtendedHarmonic
import NLsolve
using Plots
import FFTW: fft
using LinearAlgebra
using StaticArrays
using Optim

const μ₀ = 4e-7*π

include("initialize.jl")
export Ψmiller

include("shot.jl")
export Shot, psi_ρθ, plot_shot

include("FE_Fourier.jl")
export θFD_ρIP_f_nu

include("surfaces.jl")
export concentric_surface, surfaces_FE, R_Z, ρθ_RZ

end
