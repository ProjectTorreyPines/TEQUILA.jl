__precompile__()
module TEQUILA

using FiniteElementHermite
using MillerExtendedHarmonic
import NLsolve
import Plots: plot, plot!
import QuadGK: quadgk
import FFTW: fft
#using Trapz
#using BandedMatrices
#import ForwardDiff

const μ₀ = 4e-7*π

mutable struct TEQUILAEquilibrium # <: AbstractEquilibrium (eventually)
    ρ :: AbstractVector{<:Real}
    surfaces :: AbstractVector{<:MXH}
    C :: AbstractMatrix{<:Real}
end

include("initialize.jl")
export Ψmiller, first_shot

end
