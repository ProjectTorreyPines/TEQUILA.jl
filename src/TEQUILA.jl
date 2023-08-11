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
import QuadGK:quadgk
import Roots
using PreallocationTools
using JuMP
import NLopt
import BSON

const halfpi = 0.5 * π
const twopi = 2π
const μ₀ = 4e-7*π
const lower_2 = zeros(2)
const upper_2 = zeros(2)
const x0_2 = zeros(2)
const jump_success = @SVector[JuMP.OPTIMAL, JuMP.LOCALLY_SOLVED]
const int_order = 5

mutable struct QuadInfo{VR1<:AbstractVector{<:Real}, VSV1<:Vector{<:AbstractSparseVector},
                        MR1<:AbstractMatrix{<:Real}, VR2<:AbstractVector{<:Real},
                        MR2<:AbstractMatrix{<:Real}, MR3<:AbstractMatrix{<:Real}}
    x :: VR1
    w :: VR1
    νo :: VSV1
    νe :: VSV1
    D_νo :: VSV1
    D_νe :: VSV1
    R0 :: VR1
    Z0 :: VR1
    ϵ  :: VR1
    κ  :: VR1
    c0 :: VR1
    c  :: MR1
    s  :: MR1
    dR0 :: VR1
    dZ0 :: VR1
    dϵ  :: VR1
    dκ  :: VR1
    dc0 :: VR1
    dc  :: MR1
    ds  :: MR1
    θ  :: VR2
    Fsin :: MR2
    Fcos :: MR2
    gρρ :: MR3
    gρθ :: MR3
    gθθ :: MR3
end

const ProfType = Union{Nothing, FE_rep, Function}
const IpType = Union{Nothing, Real}

mutable struct Shot{I1<:Integer, VR1<:AbstractVector{<:Real}, MR1<:AbstractMatrix{<:Real}, MR2<:AbstractMatrix{<:Real},
                    PT1<:ProfType, PT2<:ProfType, PT3<:ProfType, PT4<:ProfType, PT5<:ProfType,
                    R1<:Real, R2<:Real, IP1<:IpType,
                    FE1<:FE_rep, VFE1<:AbstractVector{<:FE_rep}, Q1<:QuadInfo, VDC1<:Vector{<:DiffCache},
                    F1<:Factorization}  <: AbstractEquilibrium
    N :: I1
    M :: I1
    ρ :: VR1
    surfaces :: MR1
    C :: MR2
    P :: PT1
    dP_dψ :: PT2
    F_dF_dψ :: PT3
    Jt_R :: PT4
    Jt :: PT5
    Pbnd :: R1
    Fbnd :: R2
    Ip_target :: IP1
    R0fe::FE1
    Z0fe::FE1
    ϵfe::FE1
    κfe::FE1
    c0fe::FE1
    cfe :: VFE1
    sfe :: VFE1
    Q :: Q1
    _cx :: VDC1
    _sx :: VDC1
    _dcx :: VDC1
    _dsx :: VDC1
    _Afac :: F1
end

include("initialize.jl")
export Ψmiller

include("shot.jl")
export Shot, psi_ρθ, plot_shot, find_axis

include("quadrature.jl")
export QuadInfo

include("FE_Fourier.jl")
export θFD_ρIP_f_nu, fourier_decompose!, fft_prealloc

include("surfaces.jl")
export concentric_surface, concentric_surface!, surfaces_FE, R_Z, ρθ_RZ, surface_bracket
export Jacobian, gρρ, dR_dρ, dR_dθ, dZ_dρ, dZ_dθ

include("fit_MXH.jl")
export refit!

include("GS.jl")
export preallocate_Astar, define_Astar, define_Astar!, set_bc!, define_B, define_B!

include("fsa.jl")
export FSA, Vprime, fsa_invR2, fsa_invR, Ip

include("solve.jl")
export solve

end
