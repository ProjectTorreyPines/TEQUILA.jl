#= VEQ-style direct fixed-boundary Grad-Shafranov solve.

Solves for radial Chebyshev profiles of MXH shape coefficients (h, v, k, c0,
c_m, s_m) plus normalized-flux (psin) / toroidal-field (F) profiles by
root-finding the variationally projected strong-form GS residual.
Formulation notes and veqpy source references: docs/veq_formulation.md.

Supported so far: route :PF with coordinate :psin, nodes :uniform
(psin active family), with optional Ip constraint.

Ported from veqpy (https://github.com/zhangtakeda/veqpy),
Copyright (c) 2026 rhzhang, licensed under BSD 3-Clause.
See THIRD_PARTY_LICENSES for the full license text.
=#

const VEQ_FIX_RHO = 0.05
const VEQ_PSIN_R_FLOOR = 1.0e-10
const VEQ_J_FLOOR = 1.0e-6

# ------------------------------------------------------------------
# Topology and layout
# ------------------------------------------------------------------

"""
    VEQTopology(; h_count, v_count=0, kappa_count, c0_count=0, psin_count,
                  F_count=0, c_counts=Int[], s_counts=Int[], Nr, Nt,
                  route=:PF, coordinate=:psin, nodes=:uniform,
                  ip_constraint=false, beta_constraint=false,
                  sample_count=51, K_max=nothing)

Static solve topology: number of radial Chebyshev coefficients per active
profile family, quadrature grid size, and source route. `c_counts`/`s_counts`
are m=1-started (like `KernelTopology.s_counts` in veqpy; note veqpy's
c_counts are c0-started instead — here c0 has its own `c0_count`).
"""
struct VEQTopology
    h_count::Int
    v_count::Int
    kappa_count::Int
    c0_count::Int
    psin_count::Int
    F_count::Int
    c_counts::Vector{Int}
    s_counts::Vector{Int}
    Nr::Int
    Nt::Int
    route::Symbol
    coordinate::Symbol
    nodes::Symbol
    ip_constraint::Bool
    beta_constraint::Bool
    sample_count::Int
    K_max::Union{Nothing,Int}
end

function VEQTopology(;
    h_count::Int,
    v_count::Int=0,
    kappa_count::Int,
    c0_count::Int=0,
    psin_count::Int,
    F_count::Int=0,
    c_counts::AbstractVector{<:Integer}=Int[],
    s_counts::AbstractVector{<:Integer}=Int[],
    Nr::Int,
    Nt::Int,
    route::Symbol=:PF,
    coordinate::Symbol=:psin,
    nodes::Symbol=:uniform,
    ip_constraint::Bool=false,
    beta_constraint::Bool=false,
    sample_count::Int=51,
    K_max::Union{Nothing,Int}=nothing,
)
    @assert Nr >= 4 && Nt >= 4
    route === :PF || error("VEQ: only route=:PF implemented so far")
    (coordinate === :psin && nodes === :uniform) ||
        error("VEQ: only coordinate=:psin, nodes=:uniform implemented so far")
    psin_count > 0 || error("VEQ: psin family must be active for coordinate=:psin, nodes=:uniform")
    F_count == 0 || error("VEQ: active F family not implemented for route=:PF")
    ip_constraint && beta_constraint && error("VEQ: Ip and beta constraints are mutually exclusive for PF")
    return VEQTopology(
        h_count, v_count, kappa_count, c0_count, psin_count, F_count,
        collect(Int, c_counts), collect(Int, s_counts),
        Nr, Nt, route, coordinate, nodes, ip_constraint, beta_constraint,
        sample_count, K_max)
end

# Profile family blocks in canonical packed order (h, v, k, c0, c_m..., s_m..., psin, F)
struct VEQBlock
    name::Symbol
    kind::Symbol         # :h, :v, :k, :c0, :c, :s, :psin, :F
    order::Int           # Fourier order m for :c/:s, otherwise 0
    count::Int           # number of Chebyshev coefficients (0 = passive)
    power::Int           # rho regularity prefactor exponent
    amplitude_power::Float64
    xind::Vector{Int}    # indices into packed x (empty when passive)
end

veq_K_value(m::Int, K_max::Union{Nothing,Int}) = K_max === nothing ? m : min(m, K_max)

"""
    veq_blocks(top::VEQTopology)

Build the ordered profile blocks with packed-x indices using veqpy's "degree"
layout: coefficients interleaved round-robin over active blocks by Chebyshev
degree.
"""
function veq_blocks(top::VEQTopology)
    blocks = VEQBlock[]
    push!(blocks, VEQBlock(:h, :h, 0, top.h_count, 0, 1.0, Int[]))
    push!(blocks, VEQBlock(:v, :v, 0, top.v_count, 0, 1.0, Int[]))
    push!(blocks, VEQBlock(:k, :k, 0, top.kappa_count, 0, 1.0, Int[]))
    push!(blocks, VEQBlock(:c0, :c0, 0, top.c0_count, 0, 1.0, Int[]))
    for (m, cnt) in enumerate(top.c_counts)
        push!(blocks, VEQBlock(Symbol(:c, m), :c, m, cnt, veq_K_value(m, top.K_max), 1.0, Int[]))
    end
    for (m, cnt) in enumerate(top.s_counts)
        push!(blocks, VEQBlock(Symbol(:s, m), :s, m, cnt, veq_K_value(m, top.K_max), 1.0, Int[]))
    end
    push!(blocks, VEQBlock(:psin, :psin, 0, top.psin_count, 2, 1.0, Int[]))
    push!(blocks, VEQBlock(:F, :F, 0, top.F_count, 0, 0.5, Int[]))

    # "degree" layout: round-robin by Chebyshev degree over active blocks
    active = [b for b in blocks if b.count > 0]
    maxdeg = maximum(b.count for b in active)
    idx = 0
    for deg in 0:(maxdeg - 1)
        for b in active
            if b.count > deg
                push!(b.xind, idx += 1)
            end
        end
    end
    return blocks
end

veq_x_size(top::VEQTopology) = sum(b.count for b in veq_blocks(top))

# ------------------------------------------------------------------
# Quadrature grid, spectral operators, basis tables
# ------------------------------------------------------------------

"""Gauss-Legendre nodes/weights on (0, 1) via Golub-Welsch."""
function veq_legendre_01(n::Int)
    β = [k / sqrt(4.0 * k^2 - 1.0) for k in 1:(n - 1)]
    E = eigen(SymTridiagonal(zeros(n), β))
    nodes = E.values
    weights = [2.0 * E.vectors[1, j]^2 for j in 1:n]
    return 0.5 .* (nodes .+ 1.0), 0.5 .* weights
end

"""Legendre Vandermonde with columns P_0..P_deg evaluated at x (in [-1,1])."""
function veq_legvander(x::AbstractVector{<:Real}, deg::Int)
    V = zeros(length(x), deg + 1)
    V[:, 1] .= 1.0
    deg >= 1 && (V[:, 2] .= x)
    for k in 1:(deg - 1)
        @. V[:, k + 2] = ((2k + 1) * x * V[:, k + 1] - k * V[:, k]) / (k + 1)
    end
    return V
end

"""Barycentric spectral differentiation matrix on arbitrary nodes."""
function veq_spectral_differentiator(nodes::AbstractVector{<:Real})
    n = length(nodes)
    signs = ones(n)
    logw = zeros(n)
    for i in 1:n, j in 1:n
        i == j && continue
        d = nodes[i] - nodes[j]
        signs[i] *= sign(d)
        logw[i] -= log(abs(d))
    end
    D = zeros(n, n)
    for i in 1:n, j in 1:n
        i == j && continue
        D[i, j] = signs[i] * signs[j] * exp(clamp(logw[j] - logw[i], -700.0, 700.0)) / (nodes[i] - nodes[j])
    end
    for i in 1:n
        D[i, i] = -sum(D[i, j] for j in 1:n if j != i)
    end
    return D
end

"""Spectral prefix-integral matrix: (A*f)[i] ≈ ∫₀^{ρᵢ} f dρ for nodes in [0,1]."""
function veq_spectral_accumulator(nodes::AbstractVector{<:Real})
    n = length(nodes)
    xg = 2.0 .* nodes .- 1.0
    Vfull = veq_legvander(xg, n)          # columns P_0..P_n
    V = Vfull[:, 1:n]
    anti = zeros(n, n)
    anti[:, 1] .= 0.5 .* (xg .+ 1.0)
    for k in 1:(n - 1)
        @. anti[:, k + 1] = 0.5 * (Vfull[:, k + 2] - Vfull[:, k]) / (2k + 1)
    end
    Vlow = veq_legvander([-1.0], n)[1, :]
    low = zeros(n)
    for k in 1:(n - 1)
        low[k + 1] = 0.5 * (Vlow[k + 2] - Vlow[k]) / (2k + 1)
    end
    return ((V') \ (anti .- low')')'
end

"""Chebyshev tables T_l(x(ρ)) with ρ-derivatives; x = 2ρ²−1, rows l=0..L."""
function veq_chebyshev_tables(rho::AbstractVector{<:Real}, L_max::Int)
    n = length(rho)
    T = zeros(L_max + 1, n); Tx = zeros(L_max + 1, n); Txx = zeros(L_max + 1, n)
    x = @. 2.0 * rho^2 - 1.0
    T[1, :] .= 1.0
    if L_max >= 1
        T[2, :] .= x
        Tx[2, :] .= 1.0
    end
    for l in 2:L_max
        @. T[l + 1, :] = 2.0 * x * T[l, :] - T[l - 1, :]
        @. Tx[l + 1, :] = 2.0 * T[l, :] + 2.0 * x * Tx[l, :] - Tx[l - 1, :]
        @. Txx[l + 1, :] = 4.0 * Tx[l, :] + 2.0 * x * Txx[l, :] - Txx[l - 1, :]
    end
    T_r = similar(T); T_rr = similar(T)
    for l in 0:L_max, i in 1:n
        dxdr = 4.0 * rho[i]
        T_r[l + 1, i] = Tx[l + 1, i] * dxdr
        T_rr[l + 1, i] = Txx[l + 1, i] * dxdr^2 + Tx[l + 1, i] * 4.0
    end
    return T, T_r, T_rr
end

struct VEQGrid
    rho::Vector{Float64}
    w::Vector{Float64}
    theta::Vector{Float64}
    y::Vector{Float64}        # edge envelope 1 − ρ²
    rho_pow::Matrix{Float64}  # row p+1 = ρ^p, powers 0..maxpow
    T::Matrix{Float64}
    T_r::Matrix{Float64}
    T_rr::Matrix{Float64}
    cos_mt::Matrix{Float64}   # (M_max+1) x Nt, row m+1 = cos(mθ)
    sin_mt::Matrix{Float64}
    D::Matrix{Float64}
    A::Matrix{Float64}
    n_axis_fix::Int
end

function VEQGrid(top::VEQTopology)
    blocks = veq_blocks(top)
    L_max = maximum(b.count for b in blocks) - 1
    M_max = max(1, length(top.c_counts), length(top.s_counts))
    rho, w = veq_legendre_01(top.Nr)
    theta = [2π * j / top.Nt for j in 0:(top.Nt - 1)]
    y = @. 1.0 - rho^2
    maxpow = 2
    for b in blocks
        (b.kind === :c || b.kind === :s) && (maxpow = max(maxpow, b.power + 1))
    end
    rho_pow = [ρ^p for p in 0:maxpow, ρ in rho]
    T, T_r, T_rr = veq_chebyshev_tables(rho, L_max)
    cos_mt = [cos(m * t) for m in 0:M_max, t in theta]
    sin_mt = [sin(m * t) for m in 0:M_max, t in theta]
    D = veq_spectral_differentiator(rho)
    A = veq_spectral_accumulator(rho)
    n_fix = count(<(VEQ_FIX_RHO), rho)
    return VEQGrid(rho, w, theta, y, rho_pow, T, T_r, T_rr, cos_mt, sin_mt, D, A, n_fix)
end

# ------------------------------------------------------------------
# Case inputs
# ------------------------------------------------------------------

"""
    VEQBoundary(; a, R0, Z0=0.0, B0, ka, c0=0.0, c_offsets=Float64[], s_offsets=Float64[])

Boundary MXH parameters (edge values of the shape profiles) in veqpy's
convention: R = R0 + a(h + ρ cos θb), Z = Z0 + a(v − ρ k sin θ),
θb = θ + c0 + Σ c_m cos(mθ) + s_m sin(mθ). `c_offsets`/`s_offsets` m=1-started.
"""
Base.@kwdef struct VEQBoundary
    a::Float64
    R0::Float64
    Z0::Float64 = 0.0
    B0::Float64
    ka::Float64
    c0::Float64 = 0.0
    c_offsets::Vector{Float64} = Float64[]
    s_offsets::Vector{Float64} = Float64[]
end

"""
    VEQSource(; heat_profile, current_profile, Ip=NaN, beta=NaN)

PF-route source: `heat_profile` (pressure-gradient-like) and
`current_profile` (FF'-like) sampled on a uniform normalized-flux axis with
`top.sample_count` points. μ0 scaling of the pressure-like input happens at
materialization, mirroring veqpy.
"""
Base.@kwdef struct VEQSource
    heat_profile::Vector{Float64}
    current_profile::Vector{Float64}
    Ip::Float64 = NaN
    beta::Float64 = NaN
end

"""Not-a-knot cubic spline coefficients on a uniform grid (veqpy-compatible)."""
function veq_spline_coefficients(samples::AbstractVector{<:Real})
    n = length(samples)
    n >= 4 || error("VEQ source spline needs at least 4 samples")
    h = 1.0 / (n - 1)
    M = zeros(n, n)
    rhs = zeros(n)
    M[1, 1] = 1.0; M[1, 2] = -2.0; M[1, 3] = 1.0
    M[n, n - 2] = 1.0; M[n, n - 1] = -2.0; M[n, n] = 1.0
    for r in 2:(n - 1)
        M[r, r - 1] = 1.0; M[r, r] = 4.0; M[r, r + 1] = 1.0
        rhs[r] = 6.0 * (samples[r + 1] - 2.0 * samples[r] + samples[r - 1]) / h^2
    end
    sec = M \ rhs
    coeff = zeros(n - 1, 4)
    for iv in 1:(n - 1)
        coeff[iv, 1] = samples[iv]
        coeff[iv, 2] = samples[iv + 1] - samples[iv] - h^2 * (2.0 * sec[iv] + sec[iv + 1]) / 6.0
        coeff[iv, 3] = 0.5 * h^2 * sec[iv]
        coeff[iv, 4] = h^2 * (sec[iv + 1] - sec[iv]) / 6.0
    end
    return coeff
end

"""Evaluate uniform-interval cubic coefficients at query q ∈ [0,1] (clamped)."""
function veq_spline_eval(coeff::AbstractMatrix{Float64}, q::Real)
    nint = size(coeff, 1)
    qc = clamp(q, zero(q), one(q))
    pos = qc * nint
    # interval selection is discrete; use the primal value so Duals flow only
    # through the local coordinate t
    iv = min(floor(Int, ForwardDiff.value(pos)), nint - 1)   # q==1 stays in last interval
    t = pos - iv
    @inbounds return ((coeff[iv + 1, 4] * t + coeff[iv + 1, 3]) * t + coeff[iv + 1, 2]) * t + coeff[iv + 1, 1]
end

# ------------------------------------------------------------------
# Preallocated per-eltype workspace (cached per Dual type for ForwardDiff)
# ------------------------------------------------------------------

struct VEQWork{T}
    prof::Vector{Matrix{T}}      # per block: 3 x Nr (value, dρ, dρρ)
    R::Matrix{T}
    R_t::Matrix{T}
    Z_t::Matrix{T}
    J::Matrix{T}
    JdivR::Matrix{T}
    sin_tb::Matrix{T}
    gttdivJR::Matrix{T}
    gttdivJR_r::Matrix{T}
    grtdivJR_t::Matrix{T}
    G::Matrix{T}
    GpR::Matrix{T}
    GpZ::Matrix{T}
    GpRs::Matrix{T}
    V_r::Vector{T}
    Kn::Vector{T}
    Ln_r::Vector{T}
    psin::Vector{T}
    psin_r::Vector{T}
    psin_rr::Vector{T}
    heat::Vector{T}
    curr::Vector{T}
    integrand::Vector{T}
    t_r::Vector{T}
    Pn::Vector{T}
    FFn::Vector{T}
    collapsed::Vector{T}
end

function VEQWork{T}(nblocks::Int, Nr::Int, Nt::Int) where {T}
    mat() = Matrix{T}(undef, Nr, Nt)
    vec() = Vector{T}(undef, Nr)
    return VEQWork{T}(
        [Matrix{T}(undef, 3, Nr) for _ in 1:nblocks],
        mat(), mat(), mat(), mat(), mat(), mat(), mat(), mat(), mat(),
        mat(), mat(), mat(), mat(),
        vec(), vec(), vec(), vec(), vec(), vec(), vec(), vec(), vec(), vec(),
        vec(), vec(), vec())
end

# ------------------------------------------------------------------
# Kernel: precomputed static data for one topology + case
# ------------------------------------------------------------------

struct VEQKernel
    top::VEQTopology
    grid::VEQGrid
    blocks::Vector{VEQBlock}
    boundary::VEQBoundary
    source::VEQSource
    heat_coeff::Matrix{Float64}   # spline of μ0-scaled heat samples
    curr_coeff::Matrix{Float64}
    x_size::Int
    work::Dict{DataType,Any}      # VEQWork{T} cache; makes residual non-thread-safe
end

function VEQKernel(top::VEQTopology, boundary::VEQBoundary, source::VEQSource)
    length(source.heat_profile) == top.sample_count ||
        error("heat_profile must have sample_count=$(top.sample_count) samples")
    length(source.current_profile) == top.sample_count ||
        error("current_profile must have sample_count=$(top.sample_count) samples")
    blocks = veq_blocks(top)
    grid = VEQGrid(top)
    heat_coeff = veq_spline_coefficients(μ₀ .* source.heat_profile)
    curr_coeff = veq_spline_coefficients(source.current_profile)
    return VEQKernel(top, grid, blocks, boundary, source, heat_coeff, curr_coeff,
        sum(b.count for b in blocks), Dict{DataType,Any}())
end

function veq_getwork(kern::VEQKernel, ::Type{T}) where {T}
    return get!(kern.work, T) do
        VEQWork{T}(length(kern.blocks), kern.top.Nr, kern.top.Nt)
    end::VEQWork{T}
end

veq_offset(b::VEQBlock, bd::VEQBoundary) =
    b.kind === :k ? bd.ka :
    b.kind === :c0 ? bd.c0 :
    b.kind === :c ? (b.order <= length(bd.c_offsets) ? bd.c_offsets[b.order] : 0.0) :
    b.kind === :s ? (b.order <= length(bd.s_offsets) ? bd.s_offsets[b.order] : 0.0) :
    b.kind === :psin ? 1.0 :
    b.kind === :F ? 1.0 : 0.0

veq_scale(b::VEQBlock, bd::VEQBoundary) = b.kind === :F ? bd.R0 * bd.B0 : 1.0

# ------------------------------------------------------------------
# Stage A: profiles (value, dρ, dρρ at radial nodes)
# ------------------------------------------------------------------

"""Fill `out` (3 x Nr: value, dρ, dρρ) for one profile block."""
function veq_profile!(out::AbstractMatrix, b::VEQBlock, x::AbstractVector, kern::VEQKernel)
    g = kern.grid
    n = length(g.rho)
    offset = veq_offset(b, kern.boundary)
    scale = veq_scale(b, kern.boundary)
    T0 = eltype(out)
    @inbounds for i in 1:n
        ρ = g.rho[i]
        # Chebyshev series with edge envelope y = 1 − ρ²
        series = zero(T0); series_r = zero(T0); series_rr = zero(T0)
        for (l, xi) in enumerate(b.xind)
            c = x[xi]
            series += c * g.T[l, i]
            series_r += c * g.T_r[l, i]
            series_rr += c * g.T_rr[l, i]
        end
        env = g.y[i]
        base = env * series
        base_r = -2.0 * ρ * series + env * series_r
        base_rr = -2.0 * series - 4.0 * ρ * series_r + env * series_rr
        amp = offset + base
        amp_r = base_r
        amp_rr = base_rr
        if b.amplitude_power == 0.5
            araw = max(amp, 1.0e-10)
            av = sqrt(araw)
            amp_r = 0.5 * base_r / av
            amp_rr = 0.5 * base_rr / av - 0.25 * base_r^2 / (av * araw)
            amp = av
        end
        k = b.power
        rp = g.rho_pow[k + 1, i]
        rp_r = k == 0 ? 0.0 : k * g.rho_pow[k, i]
        rp_rr = k <= 1 ? 0.0 : k * (k - 1) * g.rho_pow[k - 1, i]
        out[1, i] = scale * rp * amp
        out[2, i] = scale * (rp_r * amp + rp * amp_r)
        out[3, i] = scale * (rp_rr * amp + 2.0 * rp_r * amp_r + rp * amp_rr)
    end
    return out
end

# ------------------------------------------------------------------
# Axis regularization helpers (veqpy abi/source_semantics.py)
# ------------------------------------------------------------------

"""Extrapolate p/ρ linearly in ρ² from anchors n_fix, n_fix+1 back to head samples."""
function veq_regularize_axis_linear!(p::AbstractVector, rho::Vector{Float64}, n_fix::Int)
    n_fix < 1 && return p
    a0, a1 = n_fix + 1, n_fix + 2
    x0, x1 = rho[a0]^2, rho[a1]^2
    r0 = p[a0] / rho[a0]
    r1 = p[a1] / rho[a1]
    grad = (r1 - r0) / (x1 - x0)
    for i in 1:n_fix
        p[i] = rho[i] * (r0 + grad * (rho[i]^2 - x0))
    end
    return p
end

"""Extrapolate p linearly in ρ² (even profile) from anchors n_fix, n_fix+1."""
function veq_regularize_axis_even!(p::AbstractVector, rho::Vector{Float64}, n_fix::Int)
    n_fix < 1 && return p
    a0, a1 = n_fix + 1, n_fix + 2
    x0, x1 = rho[a0]^2, rho[a1]^2
    grad = (p[a1] - p[a0]) / (x1 - x0)
    v0 = p[a0]
    for i in 1:n_fix
        p[i] = v0 + grad * (rho[i]^2 - x0)
    end
    return p
end

function veq_regularize_psin_r!(p::AbstractVector, rho::Vector{Float64}, n_fix::Int)
    veq_regularize_axis_linear!(p, rho, n_fix)
    for i in eachindex(p)
        p[i] = max(p[i], VEQ_PSIN_R_FLOOR)
    end
    return p
end

# ------------------------------------------------------------------
# Residual helpers (top-level functions: no closures, no boxing)
# ------------------------------------------------------------------

function veq_rowsum!(dest::AbstractVector, M::AbstractMatrix)
    @inbounds for i in axes(M, 1)
        s = zero(eltype(M))
        for j in axes(M, 2)
            s += M[i, j]
        end
        dest[i] = s
    end
    return dest
end

function veq_rowsum_w!(dest::AbstractVector, M::AbstractMatrix, wt::AbstractVector)
    @inbounds for i in axes(M, 1)
        s = zero(eltype(M))
        for j in axes(M, 2)
            s += M[i, j] * wt[j]
        end
        dest[i] = s
    end
    return dest
end

"""Project θ-collapsed residual onto the block's Chebyshev test functions."""
function veq_project!(out::AbstractVector, b::VEQBlock, T::Matrix{Float64},
    collapsed::AbstractVector, radial::AbstractVector, y::Vector{Float64},
    w::Vector{Float64}, scalar::Float64)
    @inbounds for (l, xi) in enumerate(b.xind)
        acc = zero(eltype(collapsed))
        for i in eachindex(collapsed)
            acc += T[l, i] * collapsed[i] * radial[i] * y[i] * w[i]
        end
        out[xi] = acc * scalar
    end
    return out
end

# ------------------------------------------------------------------
# Residual (fused stages A-D)
# ------------------------------------------------------------------

"""
    veq_residual!(out, x, kern::VEQKernel)

Fused profile/geometry/source/residual evaluation for packed state `x`.
Generic in eltype for ForwardDiff (workspaces are cached per eltype, so this
is not thread-safe for concurrent calls on one kernel).
Returns `(out, α1, α2)`.
"""
function veq_residual!(out::AbstractVector, x::AbstractVector, kern::VEQKernel)
    TT = promote_type(eltype(out), eltype(x))
    wk = veq_getwork(kern, TT)
    α1, α2 = veq_residual_core!(out, x, kern, wk)
    return out, α1, α2
end

function veq_residual(x::AbstractVector, kern::VEQKernel)
    out = Vector{promote_type(Float64, eltype(x))}(undef, kern.x_size)
    veq_residual!(out, x, kern)
    return out
end

function veq_residual_core!(out::AbstractVector, x::AbstractVector,
    kern::VEQKernel, wk::VEQWork{TT}) where {TT}
    g = kern.grid
    bd = kern.boundary
    top = kern.top
    Nr, Nt = top.Nr, top.Nt
    a, R0, B0 = bd.a, bd.R0, bd.B0

    # blocks are in canonical order: h=1, v=2, k=3, c0=4, c_m..., s_m..., psin, F
    blocks = kern.blocks
    nc = length(top.c_counts)
    ns = length(top.s_counts)
    c_range = 5:(4 + nc)
    s_range = (5 + nc):(4 + nc + ns)
    ipsin = length(blocks) - 1

    # --- Stage A: profiles
    @inbounds for (bi, b) in enumerate(blocks)
        bi == length(blocks) && b.count == 0 && continue  # passive F unused
        veq_profile!(wk.prof[bi], b, x, kern)
    end
    ph = wk.prof[1]; pv = wk.prof[2]; pk = wk.prof[3]; pc0 = wk.prof[4]

    # --- Stage B: geometry per point + radial moments
    @inbounds for i in 1:Nr
        ρ = g.rho[i]
        h_i = ph[1, i]; h_r_i = ph[2, i]; h_rr_i = ph[3, i]
        v_r_i = pv[2, i]; v_rr_i = pv[3, i]
        k_i = pk[1, i]; k_r_i = pk[2, i]; k_rr_i = pk[3, i]
        sum_JR = zero(TT); sum_gtt = zero(TT); sum_JdivR = zero(TT)
        for j in 1:Nt
            sin_t = g.sin_mt[2, j]; cos_t = g.cos_mt[2, j]
            tb = g.theta[j] + pc0[1, i]; tb_r = pc0[2, i]; tb_rr = pc0[3, i]
            tb_t = one(TT); tb_rt = zero(TT); tb_tt = zero(TT)
            for bi in c_range
                m = blocks[bi].order
                pm = wk.prof[bi]
                cm = g.cos_mt[m + 1, j]; sm = g.sin_mt[m + 1, j]
                tb += pm[1, i] * cm; tb_r += pm[2, i] * cm; tb_rr += pm[3, i] * cm
                tb_t -= m * pm[1, i] * sm; tb_rt -= m * pm[2, i] * sm; tb_tt -= m^2 * pm[1, i] * cm
            end
            for bi in s_range
                m = blocks[bi].order
                pm = wk.prof[bi]
                cm = g.cos_mt[m + 1, j]; sm = g.sin_mt[m + 1, j]
                tb += pm[1, i] * sm; tb_r += pm[2, i] * sm; tb_rr += pm[3, i] * sm
                tb_t += m * pm[1, i] * cm; tb_rt += m * pm[2, i] * cm; tb_tt -= m^2 * pm[1, i] * sm
            end
            stb, ctb = sincos(tb)
            R_ij = R0 + a * (h_i + ρ * ctb)
            R_r = a * (h_r_i + ctb - ρ * stb * tb_r)
            R_rr = a * (h_rr_i - 2.0 * stb * tb_r - ρ * (ctb * tb_r^2 + stb * tb_rr))
            R_t_ij = -a * ρ * stb * tb_t
            R_rt = -a * (stb * tb_t + ρ * (ctb * tb_r * tb_t + stb * tb_rt))
            R_tt = -a * ρ * (ctb * tb_t^2 + stb * tb_tt)
            Z_r = a * (v_r_i - (k_i + ρ * k_r_i) * sin_t)
            Z_t_ij = -a * ρ * k_i * cos_t
            Z_rr = a * (v_rr_i - (2.0 * k_r_i + ρ * k_rr_i) * sin_t)
            Z_rt = -a * (k_i + ρ * k_r_i) * cos_t
            Z_tt = a * ρ * k_i * sin_t

            J_ij = R_t_ij * Z_r - R_r * Z_t_ij
            J_ij = max(J_ij, VEQ_J_FLOOR)
            J_r = R_rt * Z_r + R_t_ij * Z_rr - R_rr * Z_t_ij - R_r * Z_rt
            J_t = R_tt * Z_r + R_t_ij * Z_rt - R_rt * Z_t_ij - R_r * Z_tt

            JR = J_ij * R_ij
            JR_r = J_r * R_ij + J_ij * R_r
            JR_t = J_t * R_ij + J_ij * R_t_ij
            grt = R_r * R_t_ij + Z_r * Z_t_ij
            grt_t = R_rt * R_t_ij + R_r * R_tt + Z_rt * Z_t_ij + Z_r * Z_tt
            gtt = R_t_ij^2 + Z_t_ij^2
            gtt_r = 2.0 * (R_t_ij * R_rt + Z_t_ij * Z_rt)
            invJR = 1.0 / JR

            wk.R[i, j] = R_ij; wk.R_t[i, j] = R_t_ij; wk.Z_t[i, j] = Z_t_ij
            wk.J[i, j] = J_ij; wk.JdivR[i, j] = J_ij / R_ij; wk.sin_tb[i, j] = stb
            wk.gttdivJR[i, j] = gtt * invJR
            wk.gttdivJR_r[i, j] = gtt_r * invJR - gtt * JR_r * invJR^2
            wk.grtdivJR_t[i, j] = (grt_t - grt * JR_t * invJR) * invJR

            sum_JR += JR; sum_gtt += gtt * invJR; sum_JdivR += J_ij / R_ij
        end
        wk.V_r[i] = sum_JR * (2π / Nt) * 2π
        wk.Kn[i] = sum_gtt / Nt
        wk.Ln_r[i] = sum_JdivR / Nt
    end

    # --- Stage C: source (PF, psin coordinate, uniform nodes; psin family active)
    # Residual root fields come from the x psin block with axis regularization.
    pp = wk.prof[ipsin]
    @inbounds for i in 1:Nr
        wk.psin_r[i] = pp[2, i]
    end
    veq_regularize_psin_r!(wk.psin_r, g.rho, g.n_axis_fix)
    mul!(wk.psin_rr, g.D, wk.psin_r)
    mul!(wk.psin, g.A, wk.psin_r)
    let o = wk.psin[1], s = wk.psin[end] - wk.psin[1]
        abs(s) < 1e-12 && error("psin does not span a valid normalized flux interval")
        @inbounds for i in 1:Nr
            wk.psin[i] = (wk.psin[i] - o) / s
        end
        wk.psin[1] = zero(TT); wk.psin[end] = one(TT)
    end
    @inbounds for i in 1:Nr
        wk.heat[i] = veq_spline_eval(kern.heat_coeff, wk.psin[i])   # μ0-scaled
        wk.curr[i] = veq_spline_eval(kern.curr_coeff, wk.psin[i])
    end

    # FSA-derived target ψn' (used only for the α2 magnitude via c2)
    @inbounds for i in 1:Nr
        wk.integrand[i] = wk.curr[i] * wk.Ln_r[i] + wk.V_r[i] * wk.heat[i] / (4π^2)
    end
    mul!(wk.t_r, g.A, wk.integrand)
    @inbounds for i in 1:Nr
        wk.t_r[i] = -wk.t_r[i] / wk.Kn[i]
    end
    tsum = zero(TT)
    @inbounds for i in 1:Nr
        tsum += wk.t_r[i] * g.w[i]
    end
    tsign = tsum < 0.0 ? -1.0 : 1.0
    if tsign < 0.0
        @inbounds for i in 1:Nr
            wk.t_r[i] = -wk.t_r[i]
        end
    end
    veq_regularize_psin_r!(wk.t_r, g.rho, g.n_axis_fix)
    c2 = zero(TT)
    @inbounds for i in 1:Nr
        c2 += wk.t_r[i] * g.w[i]
    end
    @inbounds for i in 1:Nr
        wk.t_r[i] /= c2
    end

    has_Ip = !isnan(kern.source.Ip)
    has_beta = !isnan(kern.source.beta)
    α1 = zero(TT)
    α2 = zero(TT)
    if !has_Ip && !has_beta
        α2 = tsign * c2
        acc = zero(TT)
        @inbounds for i in 1:Nr
            acc += wk.heat[i] * wk.t_r[i] * g.w[i]
        end
        α1 = -acc
        @inbounds for i in 1:Nr
            wk.Pn[i] = wk.heat[i] / α1
            wk.FFn[i] = wk.curr[i] / α1
        end
        veq_regularize_axis_even!(wk.FFn, g.rho, g.n_axis_fix)
    elseif has_Ip && !has_beta
        @inbounds for i in 1:Nr
            wk.Pn[i] = wk.heat[i]
            wk.FFn[i] = wk.curr[i]
        end
        veq_regularize_axis_even!(wk.FFn, g.rho, g.n_axis_fix)
        G1n_integral = zero(TT)
        @inbounds for i in 1:Nr
            G1n_integral += g.w[i] * (2π * wk.Ln_r[i] * wk.FFn[i] + wk.V_r[i] * wk.Pn[i] / (2π))
        end
        # veqpy materializes current-like constraints with μ0 scaling
        α1 = -μ₀ * kern.source.Ip / G1n_integral
        α2 = c2 * α1
    else
        error("VEQ: beta constraint not implemented yet")
    end

    # --- Stage D: residual fields and variational projections
    @inbounds for i in 1:Nr, j in 1:Nt
        invJ = 1.0 / wk.J[i, j]
        ψR = -wk.Z_t[i, j] * invJ * wk.psin_r[i]
        ψZ = wk.R_t[i, j] * invJ * wk.psin_r[i]
        G1 = wk.JdivR[i, j] * (wk.FFn[i] + wk.R[i, j]^2 * wk.Pn[i])
        G2 = wk.gttdivJR[i, j] * wk.psin_rr[i] +
             (wk.gttdivJR_r[i, j] - wk.grtdivJR_t[i, j]) * wk.psin_r[i]
        Gij = α1 * G1 + α2 * G2
        wk.G[i, j] = Gij
        wk.GpR[i, j] = Gij * ψR
        wk.GpZ[i, j] = Gij * ψZ
        wk.GpRs[i, j] = Gij * ψR * wk.sin_tb[i, j]
    end

    base = 2π / Nt
    ones_r = @view g.rho_pow[1, :]
    rho1 = @view g.rho_pow[2, :]
    rho2 = @view g.rho_pow[3, :]
    sin_t_row = @view g.sin_mt[2, :]
    for b in blocks
        b.count == 0 && continue
        if b.kind === :h
            veq_rowsum!(wk.collapsed, wk.GpR)
            veq_project!(out, b, g.T, wk.collapsed, ones_r, g.y, g.w, base * a)
        elseif b.kind === :v
            veq_rowsum!(wk.collapsed, wk.GpZ)
            veq_project!(out, b, g.T, wk.collapsed, ones_r, g.y, g.w, base * a)
        elseif b.kind === :k
            veq_rowsum_w!(wk.collapsed, wk.GpZ, sin_t_row)
            veq_project!(out, b, g.T, wk.collapsed, rho1, g.y, g.w, -base * a)
        elseif b.kind === :c0
            veq_rowsum!(wk.collapsed, wk.GpRs)
            veq_project!(out, b, g.T, wk.collapsed, rho1, g.y, g.w, -base * a)
        elseif b.kind === :c
            veq_rowsum_w!(wk.collapsed, wk.GpRs, @view g.cos_mt[b.order + 1, :])
            veq_project!(out, b, g.T, wk.collapsed,
                (@view g.rho_pow[b.power + 2, :]), g.y, g.w, -base * a)
        elseif b.kind === :s
            veq_rowsum_w!(wk.collapsed, wk.GpRs, @view g.sin_mt[b.order + 1, :])
            veq_project!(out, b, g.T, wk.collapsed,
                (@view g.rho_pow[b.power + 2, :]), g.y, g.w, -base * a)
        elseif b.kind === :psin
            veq_rowsum!(wk.collapsed, wk.G)
            veq_project!(out, b, g.T, wk.collapsed, rho2, g.y, g.w, base)
        elseif b.kind === :F
            # F block projects G against y²·T with (R0·B0)² scale; the radial
            # slot carries the second envelope factor y.
            veq_rowsum!(wk.collapsed, wk.G)
            veq_project!(out, b, g.T, wk.collapsed, g.y, g.y, g.w, base * (R0 * B0)^2)
        end
    end
    return α1, α2
end

# ------------------------------------------------------------------
# Nonlinear solve
# ------------------------------------------------------------------

"""
    veq_solve(kern::VEQKernel; x0=zeros(kern.x_size), tol=1e-9, maxiter=50)

Modified Newton on the packed residual with ForwardDiff Jacobian: the
LU-factorized Jacobian is reused across iterations while the residual keeps
contracting and refreshed when contraction stalls.
Returns `(x, converged, resnorm, iterations)`.
"""
function veq_solve(kern::VEQKernel;
    x0::AbstractVector{Float64}=zeros(kern.x_size),
    tol::Real=1e-9, maxiter::Int=50)

    x = copy(x0)
    r = veq_residual(x, kern)
    rn = norm(r)
    f!(out, xx) = (veq_residual!(out, xx, kern); out)
    Jcfg = ForwardDiff.JacobianConfig(f!, r, x)
    Jm = zeros(kern.x_size, kern.x_size)
    xt = similar(x)
    rt = similar(r)
    it = 0
    λ = 0.0
    local Jfac
    fresh = false
    refresh = true
    while rn > tol && it < maxiter
        it += 1
        if refresh
            ForwardDiff.jacobian!(Jm, f!, r, x, Jcfg)
            Jfac = lu!(copy(Jm); check=false)
            fresh = true
            refresh = false
            issuccess(Jfac) || (λ = max(λ, 1e-8))
        end
        δ = issuccess(Jfac) ? -(Jfac \ r) : -((Jm' * Jm + λ * LinearAlgebra.I) \ (Jm' * r))
        # backtracking line search
        step = 1.0
        accepted = false
        contraction = 1.0
        for _ in 1:12
            @. xt = x + step * δ
            veq_residual!(rt, xt, kern)
            rtn = norm(rt)
            if isfinite(rtn) && rtn < rn
                contraction = rtn / rn
                copyto!(x, xt); copyto!(r, rt); rn = rtn
                accepted = true
                λ = 0.1 * λ
                break
            end
            step *= 0.5
        end
        if accepted
            # stale Jacobian and weak contraction -> refresh next iteration
            (!fresh && (contraction > 0.2 || step < 1.0)) && (refresh = true)
            fresh = false
        else
            if fresh
                # even a fresh Jacobian failed a full backtrack: LM retry
                λ = λ == 0.0 ? 1e-6 : 10.0 * λ
                δ = -((Jm' * Jm + λ * LinearAlgebra.I) \ (Jm' * r))
                @. xt = x + δ
                veq_residual!(rt, xt, kern)
                rtn = norm(rt)
                if isfinite(rtn) && rtn < rn
                    copyto!(x, xt); copyto!(r, rt); rn = rtn
                else
                    break
                end
            else
                refresh = true
                it -= 1   # retry this iteration with a fresh Jacobian
            end
        end
    end
    return x, rn <= tol, rn, it
end

export VEQTopology, VEQBoundary, VEQSource, VEQKernel, veq_residual, veq_solve
