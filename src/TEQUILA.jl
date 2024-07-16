__precompile__()
module TEQUILA

using FiniteElementHermite
using MillerExtendedHarmonic
using FFTW
using LinearAlgebra
using StaticArrays
using Optim
using SparseArrays
import MXHEquilibrium: identify_cocos, transform_cocos, efit, AbstractEquilibrium, MXHEquilibrium
using RecipesBase
import PlotUtils: cgrad
import QuadGK: quadgk
import Roots
using PreallocationTools
using JuMP
import NLopt
import BSON
using LinearSolve
import ForwardDiff

const halfpi = 0.5 * π
const twopi = 2π
const μ₀ = 4e-7 * π
const lower_2 = zeros(2)
const upper_2 = zeros(2)
const x0_2 = zeros(2)
const jump_success = @SVector[JuMP.OPTIMAL, JuMP.LOCALLY_SOLVED]
const int_order = 5

# used in surfaces_FE for update_edge_derivatives!, as well as in fitted_surfaces
const δ_frac_2 = 0.5
const δ_frac_3 = 0.75

mutable struct QuadInfo{
    VR1<:AbstractVector{<:Real},
    VSV1<:Vector{<:AbstractSparseVector},
    MR1<:AbstractMatrix{<:Real},
    VR2<:AbstractVector{<:Real},
    MR2<:AbstractMatrix{<:Real},
    MR3<:AbstractMatrix{<:Real}
}
    x::VR1
    w::VR1
    νo::VSV1
    νe::VSV1
    D_νo::VSV1
    D_νe::VSV1
    R0::VR1
    Z0::VR1
    ϵ::VR1
    κ::VR1
    c0::VR1
    c::MR1
    s::MR1
    dR0::VR1
    dZ0::VR1
    dϵ::VR1
    dκ::VR1
    dc0::VR1
    dc::MR1
    ds::MR1
    θ::VR2
    Fsin::MR2
    Fcos::MR2
    gρρ::MR3
    gρθ::MR3
    gθθ::MR3
end

"""
    Profile{FE1<:FE_rep,PT1<:Union{FE_rep,Function}}

A profile (pressure- or current-like) versus normalized ρ,
    which stores the current finite-element representation, the original finite-element representation,
    and whether the ρ `grid` is `:poloidal` or `:toroidal`
"""
struct Profile{FE1<:FE_rep,PT1<:Union{FE_rep,Function}}
    fe::FE1
    orig::PT1
    grid::Symbol
end

const ProfType = Union{Nothing,Tuple{<:Union{FE_rep,Function},Symbol},Profile}

"""
    Shot{
        I1<:Integer,
        VR1<:AbstractVector{<:Real},
        MR1<:AbstractMatrix{<:Real},
        MR2<:AbstractMatrix{<:Real},
        PT1<:Union{Nothing,Profile},
        PT2<:Union{Nothing,Profile},
        PT3<:Union{Nothing,Profile},
        PT4<:Union{Nothing,Profile},
        PT5<:Union{Nothing,Profile},
        R1<:Real,
        R2<:Real,
        IP1<:Union{Nothing,Real},
        FE1<:FE_rep,
        VFE1<:AbstractVector{<:FE_rep},
        Q1<:QuadInfo,
        VDC1<:Vector{<:DiffCache},
        F1<:Factorization
    } <: AbstractEquilibrium

The fundamental data structure for a TEQUILA equilibrium, storing grid, flux-surface, and equilibrium information,
    as well as preallocated work arrays
`shot(R,Z)` returns the flux at point `(R,Z)`
"""
mutable struct Shot{
    I1<:Integer,
    VR1<:AbstractVector{<:Real},
    MR1<:AbstractMatrix{<:Real},
    MR2<:AbstractMatrix{<:Real},
    PT1<:Union{Nothing,Profile},
    PT2<:Union{Nothing,Profile},
    PT3<:Union{Nothing,Profile},
    PT4<:Union{Nothing,Profile},
    PT5<:Union{Nothing,Profile},
    R1<:Real,
    R2<:Real,
    IP1<:Union{Nothing,Real},
    FE1<:FE_rep,
    VFE1<:AbstractVector{<:FE_rep},
    Q1<:QuadInfo,
    VDC1<:Vector{<:DiffCache},
    F1<:Factorization
} <: AbstractEquilibrium
    N::I1
    M::I1
    ρ::VR1
    surfaces::MR1
    C::MR2
    P::PT1
    dP_dψ::PT2
    F_dF_dψ::PT3
    Jt_R::PT4
    Jt::PT5
    Pbnd::R1
    Fbnd::R2
    Ip_target::IP1
    R0fe::FE1
    Z0fe::FE1
    ϵfe::FE1
    κfe::FE1
    c0fe::FE1
    cfe::VFE1
    sfe::VFE1
    Q::Q1
    Vp::FE1
    invR::FE1
    invR2::FE1
    F::FE1
    ρtor::FE1
    _cx::VDC1
    _sx::VDC1
    _dcx::VDC1
    _dsx::VDC1
    _Afac::F1
end

include("initialize.jl")

include("profile.jl")

include("shot.jl")
export Shot, psi_ρθ, find_axis

include("quadrature.jl")

include("FE_Fourier.jl")

include("surfaces.jl")

include("fit_MXH.jl")

include("GS.jl")

include("fsa.jl")
export FSA, Vprime, fsa_invR2, fsa_invR, Ip

include("solve.jl")
export solve

const document = Dict()
document[Symbol(@__MODULE__)] = [:solve; [name for name in Base.names(@__MODULE__; all=false, imported=false) if (name != Symbol(@__MODULE__) && name != :solve]]

end
