__precompile__()
module TEQUILA

using FiniteElementHermite
using MillerExtendedHarmonic
using Plots
import FFTW: fft!
import LinearAlgebra: dot, ldiv!, factorize
using StaticArrays
using Optim
import LineSearches

const halfpi = 0.5 * π
const twopi = 2π
const μ₀ = 4e-7*π
const lower_2 = zeros(2)
const upper_2 = zeros(2)
const x0_2 = zeros(2)

include("initialize.jl")
export Ψmiller

include("shot.jl")
export Shot, psi_ρθ, plot_shot, find_axis

include("FE_Fourier.jl")
#export θFD_ρIP_f_nu, fourier_decompose

include("surfaces.jl")
export concentric_surface, concentric_surface!, surfaces_FE, R_Z, ρθ_RZ, surface_bracket

include("fit_MXH.jl")
export refit

end
