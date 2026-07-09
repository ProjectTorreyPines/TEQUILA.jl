#= Shot ↔ VEQ interoperability.

`veq_solve!(shot; ...)` runs the VEQ direct solve for a Shot's boundary and
pressure/current profiles and writes the converged equilibrium back into the
Shot (surfaces evaluated analytically at the knot flux levels, Ψ as an exact
flux function), reusing `update_shot!` for all derived fields.

Physical-units mapping (see docs/veq_formulation.md): with Ψbnd = 0 and
Ψ = ψ_s·(ψn − 1), the no-constraint PF route takes
    heat(ψn) = dP/dψn = ψ_s · dP/dψ
    curr(ψn) = d(F²/2)/dψn = ψ_s · F dF/dψ
and returns α2 = ψ_s² self-consistently, so ψ_s is converged by a small
outer fixed-point loop (each inner solve warm-started).
=#

"""Evaluate one VEQ profile block at arbitrary ρ (scalar version of `veq_profile`)."""
function veq_profile_at(b::VEQBlock, x::AbstractVector, kern::VEQKernel, ρ::Real)
    offset = veq_offset(b, kern.boundary)
    scale = veq_scale(b, kern.boundary)
    xc = 2.0 * ρ^2 - 1.0
    series = 0.0
    Tm1, Tl = zero(xc), one(xc)   # T_{l-1}, T_l starting at l = 0
    for (l, xi) in enumerate(b.xind)
        if l > 1
            Tm1, Tl = Tl, (l == 2 ? xc : 2.0 * xc * Tl - Tm1)
        end
        series += x[xi] * Tl
    end
    amp = offset + (1.0 - ρ^2) * series
    b.amplitude_power == 0.5 && (amp = sqrt(max(amp, 1.0e-10)))
    return scale * ρ^b.power * amp
end

"""Normalized flux ψn(ρ) from the converged packed state (smooth profile form)."""
veq_psin_at(x::AbstractVector, kern::VEQKernel, ρ::Real) =
    veq_profile_at(kern.blocks[findfirst(b -> b.name === :psin, kern.blocks)], x, kern, ρ)

"""Invert ψn(ρ_veq) = ψn_target for the VEQ radial label (monotone bisection+secant)."""
function veq_rho_of_psin(x::AbstractVector, kern::VEQKernel, ψn::Real)
    ψn <= 0.0 && return 0.0
    ψn >= 1.0 && return 1.0
    b = kern.blocks[findfirst(bb -> bb.name === :psin, kern.blocks)]
    f(ρ) = veq_profile_at(b, x, kern, ρ) - ψn
    return Roots.find_zero(f, (0.0, 1.0), Roots.Brent())
end

"""Extract the VEQ boundary from a Shot's boundary surface and Fbnd."""
function VEQBoundary(shot::Shot)
    bnd = @view shot.surfaces[:, end]
    R0, Z0, ϵ, κ, c0 = bnd[1], bnd[2], bnd[3], bnd[4], bnd[5]
    L = length(shot.cfe)
    c_offsets = collect(Float64, bnd[6:(5 + L)])
    s_offsets = collect(Float64, bnd[(6 + L):(5 + 2L)])
    # MillerExtendedHarmonic and VEQ share the same parameterization:
    # R = R0 + a cos(θ + c0 + Σ c cos(mθ) + s sin(mθ)), Z = Z0 − κ a sin θ
    return VEQBoundary(; a=ϵ * R0, R0, Z0, B0=shot.Fbnd / R0, ka=κ, c0,
        c_offsets, s_offsets)
end

"""Rebuild kernel with new source samples, reusing the grid, layout, and workspace cache."""
veq_with_source(kern::VEQKernel, source::VEQSource) = VEQKernel(
    kern.top, kern.grid, kern.blocks, kern.boundary, source,
    veq_spline_coefficients(μ₀ .* source.heat_profile),
    veq_spline_coefficients(source.current_profile),
    kern.x_size, kern.work)

function veq_source_samples(shot::Shot, ψ_s::Real, sample_count::Int)
    shot.dP_dψ !== nothing || error("veq_solve! requires dP_dψ (P and current-based inputs not yet supported)")
    shot.F_dF_dψ !== nothing || error("veq_solve! requires F_dF_dψ (Jt/Jt_R inputs not yet supported)")
    ψn_axis = range(0.0, 1.0, sample_count)
    heat = Vector{Float64}(undef, sample_count)
    curr = Vector{Float64}(undef, sample_count)
    for (i, ψn) in enumerate(ψn_axis)
        ρpol = sqrt(ψn)
        ρp = shot.dP_dψ.grid === :poloidal ? ρpol : shot.ρtor(ρpol)
        ρf = shot.F_dF_dψ.grid === :poloidal ? ρpol : shot.ρtor(ρpol)
        heat[i] = shot.dP_dψ.fe(ρp) * ψ_s
        curr[i] = shot.F_dF_dψ.fe(ρf) * ψ_s
    end
    return heat, curr
end

"""Evaluate the flattened MXH surface (Shot layout) at normalized flux ψn."""
function veq_flat_surface!(flat::AbstractVector, x::AbstractVector, kern::VEQKernel, ψn::Real, L::Int)
    bd = kern.boundary
    ρv = veq_rho_of_psin(x, kern, ψn)
    byname = Dict(b.name => b for b in kern.blocks)
    h = veq_profile_at(byname[:h], x, kern, ρv)
    v = veq_profile_at(byname[:v], x, kern, ρv)
    k = veq_profile_at(byname[:k], x, kern, ρv)
    c0 = veq_profile_at(byname[:c0], x, kern, ρv)
    R0k = bd.R0 + bd.a * h
    flat .= 0.0
    flat[1] = R0k
    flat[2] = bd.Z0 + bd.a * v
    flat[3] = bd.a * ρv / R0k
    flat[4] = k
    flat[5] = c0
    for b in kern.blocks
        (b.kind === :c || b.kind === :s) || continue
        b.order <= L || continue
        val = veq_profile_at(b, x, kern, ρv)
        flat[(b.kind === :c ? 5 : 5 + L) + b.order] = val
    end
    return flat
end

"""
    veq_solve!(shot::Shot;
        h_count=3, kappa_count=6, psin_count=6, c_counts=zeros(Int, length(shot.cfe)),
        s_counts=[3; zeros(Int, length(shot.sfe) - 1)],
        Nr=16, Nt=16, sample_count=51,
        outer_its=10, outer_tol=1e-8, tol=1e-9, debug=false)

Run the VEQ direct solve for `shot`'s boundary and dP_dψ/F_dF_dψ profiles and
write the result back into `shot`. The flux scale ψ_s = −Ψaxis is converged by
an outer fixed point (α2 = ψ_s²), warm-starting each inner Newton solve.

Counts arrays may include trailing zeros: those harmonics stay passive
(boundary value scaled by ρ^K_m) but still shape the geometry.

Returns `shot`.
"""
function veq_solve!(shot::Shot;
    h_count::Int=3, v_count::Int=0, kappa_count::Int=6, c0_count::Int=0, psin_count::Int=6,
    c_counts::AbstractVector{<:Integer}=zeros(Int, length(shot.cfe)),
    s_counts::AbstractVector{<:Integer}=[3; zeros(Int, length(shot.sfe) - 1)],
    Nr::Int=16, Nt::Int=16, sample_count::Int=51,
    outer_its::Int=10, outer_tol::Real=1e-8, tol::Real=1e-9,
    ψ_s0::Real=-1.0, debug::Bool=false)

    top = VEQTopology(; h_count, v_count, kappa_count, c0_count, psin_count,
        c_counts, s_counts, Nr, Nt, sample_count)
    boundary = VEQBoundary(shot)

    # flux scale ψ_s = dΨ/dψn = −Ψaxis (Ψbnd = 0). For an uninitialized Shot
    # (Ψ ≡ 0), fall back to ψ_s0, whose sign selects the flux orientation.
    _, _, Ψaxis = find_axis(shot)
    ψ_s = Ψaxis == 0.0 ? float(ψ_s0) : -Ψaxis

    heat, curr = veq_source_samples(shot, ψ_s, sample_count)
    kern = VEQKernel(top, boundary, VEQSource(; heat_profile=heat, current_profile=curr))

    # Flux-convention note: TEQUILA's Ψ is the total poloidal flux (Wb) with
    # Ψbnd = 0, while VEQ's GS residual is written for per-radian flux
    # ψ = Ψ/2π, so α2 = (dψ/dψn)² = (ψ_s/2π)². The self-consistency condition
    # is therefore α2(ψ_s) = (ψ_s/2π)². Both source inputs scale linearly with
    # ψ_s, which leaves the solved shape x invariant and makes α2 exactly
    # linear in ψ_s — so the fixed point ψ_s* = 4π²·α2(ψ_s)/ψ_s is exact after
    # a single inner solve (up to spline re-materialization effects), and the
    # loop below typically finishes in 2-3 iterations.
    x = zeros(kern.x_size)
    local ok, rn, it
    for outer in 1:outer_its
        x, ok, rn, it = veq_solve(kern; x0=x, tol)
        ok || @warn "veq_solve!: inner Newton did not reach tol" outer rn it
        _, α1, α2 = veq_residual!(similar(x), x, kern)
        ψ_s_new = 4π^2 * α2 / ψ_s
        debug && println("outer $outer: ψ_s = $ψ_s → $ψ_s_new, inner its = $it, |r| = $rn")
        dψ = abs(ψ_s_new - ψ_s) / abs(ψ_s_new)
        ψ_s = ψ_s_new
        heat, curr = veq_source_samples(shot, ψ_s, sample_count)
        kern = veq_with_source(kern, VEQSource(; heat_profile=heat, current_profile=curr))
        dψ < outer_tol && break
    end

    # --- write back: surfaces at the Shot's flux knots + exact flux-function Ψ
    L = length(shot.cfe)
    surfaces = similar(shot.surfaces)
    for kknot in eachindex(shot.ρ)
        @views veq_flat_surface!(surfaces[:, kknot], x, kern, shot.ρ[kknot]^2, L)
    end
    δρ = shot.ρ[end] - shot.ρ[end - 1]
    flat_δ2 = zeros(2L + 5)
    veq_flat_surface!(flat_δ2, x, kern, (shot.ρ[end - 1] + δ_frac_2 * δρ)^2, L)
    flat_δ3 = zeros(2L + 5)
    veq_flat_surface!(flat_δ3, x, kern, (shot.ρ[end - 1] + δ_frac_3 * δρ)^2, L)

    update_shot!(shot, surfaces, -ψ_s, flat_δ2, flat_δ3)
    return shot
end

export veq_solve!
