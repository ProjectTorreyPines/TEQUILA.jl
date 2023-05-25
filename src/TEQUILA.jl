__precompile__()
module TEQUILA

using FiniteElementHermite
using MillerExtendedHarmonic
using FFTW
using LinearAlgebra
using StaticArrays
using Optim
using SparseArrays
using Contour
import MXHEquilibrium: identify_cocos, transform_cocos, efit, AbstractEquilibrium, MXHEquilibrium
using RecipesBase
import PlotUtils: cgrad
import QuadGK:quadgk
import Roots
#using Memoize

const halfpi = 0.5 * π
const twopi = 2π
const μ₀ = 4e-7*π
const lower_2 = zeros(2)
const upper_2 = zeros(2)
const x0_2 = zeros(2)
const Ncntr = 801
const Ψcntr = zeros(Ncntr, Ncntr)

include("initialize.jl")
export Ψmiller

include("shot.jl")
export Shot, psi_ρθ, plot_shot, find_axis

include("FE_Fourier.jl")
export θFD_ρIP_f_nu, fourier_decompose!, fft_prealloc

include("surfaces.jl")
export concentric_surface, concentric_surface!, surfaces_FE, R_Z, ρθ_RZ, surface_bracket
export Jacobian, gρρ, dR_dρ, dR_dθ, dZ_dρ, dZ_dθ

include("fit_MXH.jl")
export refit

include("GS.jl")
export preallocate_Astar, define_Astar, define_Astar!, set_bc!, define_B, define_B!

include("fsa.jl")
export FSA, Vprime, fsa_invR2, fsa_invR, Ip

include("solve.jl")
export solve

end
