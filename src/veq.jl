#= VEQ-style direct fixed-boundary Grad-Shafranov solve.

Solves for radial Chebyshev profiles of MXH shape coefficients (h, v, k, c0,
c_m, s_m) plus normalized-flux (psin) / toroidal-field (F) profiles by
root-finding the variationally projected strong-form GS residual.
Formulation notes and veqpy source references: docs/veq_formulation.md.

Supported so far: route :PF with coordinate :psin, nodes :uniform
(psin active family), with optional Ip constraint.
=#

const VEQ_FIX_RHO = 0.05
const VEQ_PSIN_R_FLOOR = 1.0e-10
const VEQ_J_FLOOR = 1.0e-6

# ------------------------------------------------------------------
# Topology and layout
# ------------------------------------------------------------------

"""
    VEQTopology(; h_count, v_count=0, kappa_count, psin_count, F_count=0,
                  c_counts=Int[], s_counts=Int[], Nr, Nt,
                  route=:PF, coordinate=:psin, nodes=:uniform,
                  ip_constraint=false, beta_constraint=false,
                  sample_count=51, K_max=nothing)

Static solve topology: number of radial Chebyshev coefficients per active
profile family, quadrature grid size, and source route. `c_counts`/`s_counts`
are m=1-started (like `KernelTopology.s_counts` in veqpy).
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
    T, T_r, T_rr = veq_chebyshev_tables(rho, L_max)
    cos_mt = [cos(m * t) for m in 0:M_max, t in theta]
    sin_mt = [sin(m * t) for m in 0:M_max, t in theta]
    D = veq_spectral_differentiator(rho)
    A = veq_spectral_accumulator(rho)
    n_fix = count(<(VEQ_FIX_RHO), rho)
    return VEQGrid(rho, w, theta, T, T_r, T_rr, cos_mt, sin_mt, D, A, n_fix)
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
    c = @view coeff[iv + 1, :]
    return ((c[4] * t + c[3]) * t + c[2]) * t + c[1]
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
        sum(b.count for b in blocks))
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

"""Evaluate one profile block; returns (val, val_r, val_rr) length-Nr vectors."""
function veq_profile(b::VEQBlock, x::AbstractVector, kern::VEQKernel, ::Type{TT}) where {TT}
    g = kern.grid
    n = length(g.rho)
    val = zeros(TT, n); val_r = zeros(TT, n); val_rr = zeros(TT, n)
    offset = veq_offset(b, kern.boundary)
    scale = veq_scale(b, kern.boundary)
    for i in 1:n
        ρ = g.rho[i]
        # Chebyshev series with edge envelope y = 1 − ρ²
        series = zero(TT); series_r = zero(TT); series_rr = zero(TT)
        for (l, xi) in enumerate(b.xind)
            c = x[xi]
            series += c * g.T[l, i]
            series_r += c * g.T_r[l, i]
            series_rr += c * g.T_rr[l, i]
        end
        env = 1.0 - ρ^2
        base = env * series
        base_r = -2.0 * ρ * series + env * series_r
        base_rr = -2.0 * series + 2.0 * (-2.0 * ρ) * series_r + env * series_rr
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
        rp = ρ^k
        rp_r = k == 0 ? 0.0 : k * ρ^(k - 1)
        rp_rr = k <= 1 ? 0.0 : k * (k - 1) * ρ^(k - 2)
        val[i] = scale * rp * amp
        val_r[i] = scale * (rp_r * amp + rp * amp_r)
        val_rr[i] = scale * (rp_rr * amp + 2.0 * rp_r * amp_r + rp * amp_rr)
    end
    return val, val_r, val_rr
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
# Residual (fused stages B, C, D)
# ------------------------------------------------------------------

"""
    veq_residual!(out, x, kern::VEQKernel)

Fused profile/geometry/source/residual evaluation for packed state `x`.
Generic in eltype for ForwardDiff. Returns `(out, α1, α2)`.
"""
function veq_residual!(out::AbstractVector, x::AbstractVector, kern::VEQKernel)
    TT = promote_type(eltype(out), eltype(x))
    g = kern.grid
    bd = kern.boundary
    top = kern.top
    Nr, Nt = top.Nr, top.Nt
    a, R0, Z0, B0 = bd.a, bd.R0, bd.Z0, bd.B0

    blocks = kern.blocks
    bidx = Dict(b.name => b for b in blocks)

    # --- Stage A: profiles
    h, h_r, h_rr = veq_profile(bidx[:h], x, kern, TT)
    v, v_r, v_rr = veq_profile(bidx[:v], x, kern, TT)
    k, k_r, k_rr = veq_profile(bidx[:k], x, kern, TT)
    c0, c0_r, c0_rr = veq_profile(bidx[:c0], x, kern, TT)
    cblocks = [b for b in blocks if b.kind === :c]
    sblocks = [b for b in blocks if b.kind === :s]
    cprof = [veq_profile(b, x, kern, TT) for b in cblocks]
    sprof = [veq_profile(b, x, kern, TT) for b in sblocks]
    psin_prof, psin_prof_r, psin_prof_rr = veq_profile(bidx[:psin], x, kern, TT)

    # --- Stage B: geometry per point + radial moments
    R = Matrix{TT}(undef, Nr, Nt); R_t = similar(R); Z_t = similar(R)
    J = similar(R); JdivR = similar(R); sin_tb = similar(R)
    gttdivJR = similar(R); gttdivJR_r = similar(R); grtdivJR_t = similar(R)
    V_r = zeros(TT, Nr); Kn = zeros(TT, Nr); Ln_r = zeros(TT, Nr)

    for i in 1:Nr
        ρ = g.rho[i]
        sum_JR = zero(TT); sum_gtt = zero(TT); sum_JdivR = zero(TT)
        for j in 1:Nt
            θ = g.theta[j]
            sin_t = g.sin_mt[2, j]; cos_t = g.cos_mt[2, j]
            tb = θ + c0[i]; tb_r = c0_r[i]; tb_rr = c0_rr[i]
            tb_t = one(TT); tb_rt = zero(TT); tb_tt = zero(TT)
            for (bm, (cv, cvr, cvrr)) in zip(cblocks, cprof)
                m = bm.order
                cm = g.cos_mt[m + 1, j]; sm = g.sin_mt[m + 1, j]
                tb += cv[i] * cm; tb_r += cvr[i] * cm; tb_rr += cvrr[i] * cm
                tb_t += -m * cv[i] * sm; tb_rt += -m * cvr[i] * sm; tb_tt += -m^2 * cv[i] * cm
            end
            for (bm, (sv, svr, svrr)) in zip(sblocks, sprof)
                m = bm.order
                cm = g.cos_mt[m + 1, j]; sm = g.sin_mt[m + 1, j]
                tb += sv[i] * sm; tb_r += svr[i] * sm; tb_rr += svrr[i] * sm
                tb_t += m * sv[i] * cm; tb_rt += m * svr[i] * cm; tb_tt += -m^2 * sv[i] * sm
            end
            stb, ctb = sincos(tb)
            R_ij = R0 + a * (h[i] + ρ * ctb)
            R_r = a * (h_r[i] + ctb - ρ * stb * tb_r)
            R_rr = a * (h_rr[i] - 2.0 * stb * tb_r - ρ * (ctb * tb_r^2 + stb * tb_rr))
            R_t_ij = -a * ρ * stb * tb_t
            R_rt = -a * (stb * tb_t + ρ * (ctb * tb_r * tb_t + stb * tb_rt))
            R_tt = -a * ρ * (ctb * tb_t^2 + stb * tb_tt)
            Z_r = a * (v_r[i] - (k[i] + ρ * k_r[i]) * sin_t)
            Z_t_ij = -a * ρ * k[i] * cos_t
            Z_rr = a * (v_rr[i] - (2.0 * k_r[i] + ρ * k_rr[i]) * sin_t)
            Z_rt = -a * (k[i] + ρ * k_r[i]) * cos_t
            Z_tt = a * ρ * k[i] * sin_t

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

            R[i, j] = R_ij; R_t[i, j] = R_t_ij; Z_t[i, j] = Z_t_ij
            J[i, j] = J_ij; JdivR[i, j] = J_ij / R_ij; sin_tb[i, j] = stb
            gttdivJR[i, j] = gtt * invJR
            gttdivJR_r[i, j] = gtt_r * invJR - gtt * JR_r * invJR^2
            grtdivJR_t[i, j] = (grt_t - grt * JR_t * invJR) * invJR

            sum_JR += JR; sum_gtt += gtt * invJR; sum_JdivR += J_ij / R_ij
        end
        V_r[i] = sum_JR * (2π / Nt) * 2π
        Kn[i] = sum_gtt / Nt
        Ln_r[i] = sum_JdivR / Nt
    end

    # --- Stage C: source (PF, psin coordinate, uniform nodes; psin family active)
    # Residual root fields come from the x psin block with axis regularization.
    psin_r = copy(psin_prof_r)
    veq_regularize_psin_r!(psin_r, g.rho, g.n_axis_fix)
    psin_rr = g.D * psin_r
    psin = g.A * psin_r
    let o = psin[1], s = psin[end] - psin[1]
        abs(s) < 1e-12 && error("psin does not span a valid normalized flux interval")
        @. psin = (psin - o) / s
        psin[1] = zero(TT); psin[end] = one(TT)
    end
    heat = [veq_spline_eval(kern.heat_coeff, psin[i]) for i in 1:Nr]   # μ0-scaled
    curr = [veq_spline_eval(kern.curr_coeff, psin[i]) for i in 1:Nr]

    # FSA-derived target ψn' (used only for the α2 magnitude via c2)
    integrand = [curr[i] * Ln_r[i] + V_r[i] * heat[i] / (4π^2) for i in 1:Nr]
    t_r = g.A * integrand
    @. t_r = -t_r / Kn
    tsign = sum(t_r[i] * g.w[i] for i in 1:Nr) < 0.0 ? -1.0 : 1.0
    tsign < 0.0 && (t_r .*= -1.0)
    veq_regularize_psin_r!(t_r, g.rho, g.n_axis_fix)
    c2 = sum(t_r[i] * g.w[i] for i in 1:Nr)
    t_r ./= c2

    has_Ip = !isnan(kern.source.Ip)
    has_beta = !isnan(kern.source.beta)
    local α1::TT, α2::TT
    local Pn::Vector{TT}, FFn::Vector{TT}
    if !has_Ip && !has_beta
        α2 = tsign * c2
        α1 = -sum(heat[i] * t_r[i] * g.w[i] for i in 1:Nr)
        Pn = heat ./ α1
        FFn = curr ./ α1
        veq_regularize_axis_even!(FFn, g.rho, g.n_axis_fix)
    elseif has_Ip && !has_beta
        Pn = copy(heat)
        FFn = copy(curr)
        veq_regularize_axis_even!(FFn, g.rho, g.n_axis_fix)
        G1n_integral = sum(g.w[i] * (2π * Ln_r[i] * FFn[i] + V_r[i] * Pn[i] / (2π)) for i in 1:Nr)
        # veqpy materializes current-like constraints with μ0 scaling
        α1 = -μ₀ * kern.source.Ip / G1n_integral
        α2 = c2 * α1
    else
        error("VEQ: beta constraint not implemented yet")
    end

    # --- Stage D: residual fields and variational projections
    G = Matrix{TT}(undef, Nr, Nt)
    Gpsin_R = similar(G); Gpsin_Z = similar(G); Gpsin_R_stb = similar(G)
    for i in 1:Nr, j in 1:Nt
        invJ = 1.0 / J[i, j]
        ψR = -Z_t[i, j] * invJ * psin_r[i]
        ψZ = R_t[i, j] * invJ * psin_r[i]
        G1 = JdivR[i, j] * (FFn[i] + R[i, j]^2 * Pn[i])
        G2 = gttdivJR[i, j] * psin_rr[i] + (gttdivJR_r[i, j] - grtdivJR_t[i, j]) * psin_r[i]
        Gij = α1 * G1 + α2 * G2
        G[i, j] = Gij
        Gpsin_R[i, j] = Gij * ψR
        Gpsin_Z[i, j] = Gij * ψZ
        Gpsin_R_stb[i, j] = Gij * ψR * sin_tb[i, j]
    end

    base = 2π / Nt
    rowsum(M) = [sum(@view M[i, :]) for i in 1:Nr]
    rowsum_w(M, wt) = [sum(M[i, j] * wt[j] for j in 1:Nt) for i in 1:Nr]

    function project!(b::VEQBlock, collapsed::AbstractVector, radial::AbstractVector, scalar::Float64)
        for (l, xi) in enumerate(b.xind)
            acc = zero(TT)
            for i in 1:Nr
                acc += g.T[l, i] * collapsed[i] * radial[i] * (1.0 - g.rho[i]^2) * g.w[i]
            end
            out[xi] = acc * scalar
        end
    end

    ones_r = ones(Nr)
    rho1 = g.rho
    for b in blocks
        b.count == 0 && continue
        if b.kind === :h
            project!(b, rowsum(Gpsin_R), ones_r, base * a)
        elseif b.kind === :v
            project!(b, rowsum(Gpsin_Z), ones_r, base * a)
        elseif b.kind === :k
            project!(b, rowsum_w(Gpsin_Z, @view g.sin_mt[2, :]), rho1, -base * a)
        elseif b.kind === :c0
            project!(b, rowsum(Gpsin_R_stb), rho1, -base * a)
        elseif b.kind === :c
            project!(b, rowsum_w(Gpsin_R_stb, @view g.cos_mt[b.order + 1, :]),
                g.rho .^ (b.power + 1), -base * a)
        elseif b.kind === :s
            project!(b, rowsum_w(Gpsin_R_stb, @view g.sin_mt[b.order + 1, :]),
                g.rho .^ (b.power + 1), -base * a)
        elseif b.kind === :psin
            project!(b, rowsum(G), g.rho .^ 2, base)
        elseif b.kind === :F
            # F block projects G against y²·T with (R0·B0)² scale; radial slot
            # carries the second envelope factor y.
            project!(b, rowsum(G), 1.0 .- rho1 .^ 2, base * (R0 * B0)^2)
        end
    end
    return out, α1, α2
end

function veq_residual(x::AbstractVector, kern::VEQKernel)
    out = Vector{promote_type(Float64, eltype(x))}(undef, kern.x_size)
    veq_residual!(out, x, kern)
    return out
end

# ------------------------------------------------------------------
# Nonlinear solve
# ------------------------------------------------------------------

"""
    veq_solve(kern::VEQKernel; x0=zeros(kern.x_size), tol=1e-9, maxiter=50)

Damped Newton on the packed residual with ForwardDiff Jacobian.
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
    it = 0
    λ = 0.0
    while rn > tol && it < maxiter
        it += 1
        ForwardDiff.jacobian!(Jm, f!, r, x, Jcfg)
        δ = try
            -(Jm \ r)
        catch
            λ = max(λ, 1e-8)
            -((Jm' * Jm + λ * I) \ (Jm' * r))
        end
        # backtracking line search
        step = 1.0
        accepted = false
        for _ in 1:12
            xt = x .+ step .* δ
            rt = veq_residual(xt, kern)
            rtn = norm(rt)
            if isfinite(rtn) && rtn < rn
                x, r, rn = xt, rt, rtn
                accepted = true
                λ = 0.1 * λ
                break
            end
            step *= 0.5
        end
        if !accepted
            # steepest-descent-flavored LM retry
            λ = λ == 0.0 ? 1e-6 : 10.0 * λ
            δ = -((Jm' * Jm + λ * I) \ (Jm' * r))
            xt = x .+ δ
            rt = veq_residual(xt, kern)
            rtn = norm(rt)
            if isfinite(rtn) && rtn < rn
                x, r, rn = xt, rt, rtn
            else
                break
            end
        end
    end
    return x, rn <= tol, rn, it
end

export VEQTopology, VEQBoundary, VEQSource, VEQKernel, veq_residual, veq_solve
