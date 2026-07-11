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

Ported from veqpy (https://github.com/zhangtakeda/veqpy),
Copyright (c) 2026 rhzhang, licensed under BSD 3-Clause.
See THIRD_PARTY_LICENSES for the full license text.
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

"""
Monotone ψn(ρ) inversion table for the writeback.

The kernel's residual uses an axis-regularized, floored ψn' (see
`veq_regularize_psin_r!`), so the raw ψn Chebyshev profile of an (even
converged) state may be locally non-monotone at the truncation-error level.
Inverting the raw profile then maps nearby knot flux levels to out-of-order
radii and produces crossing surfaces. Mirror the kernel instead: sample the
raw profile, floor its increments, and renormalize to span [0, 1].
Returns `(ρs, ψtab)` with `ψtab` strictly increasing.
"""
function veq_rho_psin_table(x::AbstractVector, kern::VEQKernel; n::Int=4001)
    b = kern.blocks[findfirst(bb -> bb.name === :psin, kern.blocks)]
    ρs = range(0.0, 1.0, n)
    ψtab = Vector{Float64}(undef, n)
    for (i, ρ) in enumerate(ρs)
        ψtab[i] = veq_profile_at(b, x, kern, ρ)
    end
    floor_dψ = VEQ_PSIN_R_FLOOR * step(ρs)
    for i in 2:n
        ψtab[i] < ψtab[i-1] + floor_dψ && (ψtab[i] = ψtab[i-1] + floor_dψ)
    end
    ψ0 = ψtab[1]
    invs = 1.0 / (ψtab[end] - ψ0)
    @. ψtab = (ψtab - ψ0) * invs
    ψtab[1] = 0.0
    ψtab[end] = 1.0
    return ρs, ψtab
end

"""Invert the monotone table from `veq_rho_psin_table` at `ψn` (linear interpolation)."""
function veq_rho_of_psin(ρs::AbstractRange, ψtab::Vector{Float64}, ψn::Real)
    ψn <= 0.0 && return 0.0
    ψn >= 1.0 && return 1.0
    j = searchsortedfirst(ψtab, ψn)
    j <= 1 && return 0.0
    t = (ψn - ψtab[j-1]) / (ψtab[j] - ψtab[j-1])
    return ρs[j-1] + t * step(ρs)
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

# NB: callers must run update_profiles! (and scale_Ip!, if constrained) first,
# in that order — update_profile! re-materializes :toroidal fe from orig, which
# would wipe a preceding rescale (same ordering as the Picard loop).
function veq_source_samples(shot::Shot, ψ_s::Real, sample_count::Int)
    Pp = Pprime(shot, shot.P, shot.dP_dψ)
    ffp = FFprime(shot, shot.F_dF_dψ, shot.Jt_R, shot.Jt; invR=shot.invR, invR2=shot.invR2)
    ψn_axis = range(0.0, 1.0, sample_count)
    heat = Vector{Float64}(undef, sample_count)
    curr = Vector{Float64}(undef, sample_count)
    for (i, ψn) in enumerate(ψn_axis)
        ρpol = sqrt(ψn)
        heat[i] = Pp(ρpol) * ψ_s
        curr[i] = ffp(ρpol) * ψ_s
    end
    return heat, curr
end

# P, Jt, and Jt_R inputs (and any :toroidal-grid profile) are converted to
# dP_dψ/F_dF_dψ through the Shot's current Ψ and flux-surface averages, so the
# outer loop must write the equilibrium back into the Shot between iterations.
# Same for an Ip constraint, which rescales against Ip(shot).
# Pure :poloidal dP_dψ/F_dF_dψ inputs scale linearly with ψ_s and need neither.
function veq_needs_geometry(shot::Shot)
    (shot.P !== nothing || shot.Jt !== nothing || shot.Jt_R !== nothing) && return true
    shot.Ip_target !== nothing && return true
    shot.dP_dψ !== nothing && shot.dP_dψ.grid === :toroidal && return true
    shot.F_dF_dψ !== nothing && shot.F_dF_dψ.grid === :toroidal && return true
    return false
end

"""Set Ψ to the exact flux-function form Ψaxis·(1−ρ²) and refresh FSAs (for uninitialized Shots)."""
function veq_init_flux!(shot::Shot, Ψaxis::Real)
    shot.C .= 0.0
    shot.C[2:2:end, 1] .= Ψaxis .* (1.0 .- shot.ρ .^ 2)
    shot.C[1:2:end, 1] .= -2.0 .* Ψaxis .* shot.ρ
    set_FSAs!(shot)
    return shot
end

# Initial flux-scale guess for an uninitialized Shot. TEQUILA's convention has
# sign(ψ_s) = sign(-Ψaxis) = sign(Ip), with magnitude ~ μ0·Ip·R0 (checked vs
# Picard). Only the sign matters for convergence of the outer loop.
function veq_guess_flux_scale(shot::Shot)
    I = shot.Ip_target !== nothing ? shot.Ip_target : Ip(shot)
    I == 0.0 && return -1.0
    R0 = shot.surfaces[1, end]
    return μ₀ * I * R0
end

"""Evaluate the flattened MXH surface (Shot layout) at normalized flux ψn (`ρv` from raw-profile inversion by default)."""
function veq_flat_surface!(flat::AbstractVector, x::AbstractVector, kern::VEQKernel, ψn::Real, L::Int;
    ρv::Real=veq_rho_of_psin(x, kern, ψn))
    bd = kern.boundary
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

"""Check that MXH knot surfaces are strictly nested (minor radius increasing, Rmax increasing, Rmin decreasing)."""
function veq_surfaces_nested(surfaces::AbstractMatrix{<:Real})
    R0 = surfaces[1, 1]
    a_prev = R0 * surfaces[3, 1]
    Rmax_prev = R0 + a_prev
    Rmin_prev = R0 - a_prev
    for k in 2:size(surfaces, 2)
        R0 = surfaces[1, k]
        a = R0 * surfaces[3, k]
        Rmax = R0 + a
        Rmin = R0 - a
        (a > a_prev && Rmax > Rmax_prev && Rmin < Rmin_prev) || return false
        a_prev, Rmax_prev, Rmin_prev = a, Rmax, Rmin
    end
    return true
end

"""
Write the VEQ state back into `shot`: MXH surfaces at the knot flux levels + exact flux-function Ψ.

An unconverged state can produce non-nested (crossing) surfaces whose flux-surface averages
are invalid; in that case the shot is left untouched and `false` is returned so the outer
loop can proceed on the previous geometry.
"""
function veq_writeback!(shot::Shot, x::AbstractVector, kern::VEQKernel, ψ_s::Real)
    L = length(shot.cfe)
    ρtab, ψtab = veq_rho_psin_table(x, kern)
    surfaces = similar(shot.surfaces)
    for kknot in eachindex(shot.ρ)
        ρv = veq_rho_of_psin(ρtab, ψtab, shot.ρ[kknot]^2)
        @views veq_flat_surface!(surfaces[:, kknot], x, kern, shot.ρ[kknot]^2, L; ρv)
    end
    veq_surfaces_nested(surfaces) || return false
    δρ = shot.ρ[end] - shot.ρ[end - 1]
    flat_δ2 = zeros(2L + 5)
    ψn_δ2 = (shot.ρ[end - 1] + δ_frac_2 * δρ)^2
    veq_flat_surface!(flat_δ2, x, kern, ψn_δ2, L; ρv=veq_rho_of_psin(ρtab, ψtab, ψn_δ2))
    flat_δ3 = zeros(2L + 5)
    ψn_δ3 = (shot.ρ[end - 1] + δ_frac_3 * δρ)^2
    veq_flat_surface!(flat_δ3, x, kern, ψn_δ3, L; ρv=veq_rho_of_psin(ρtab, ψtab, ψn_δ3))

    update_shot!(shot, surfaces, -ψ_s, flat_δ2, flat_δ3)
    return true
end

"""
    veq_solve!(shot::Shot;
        h_count=3, kappa_count=6, psin_count=6, c_counts=zeros(Int, length(shot.cfe)),
        s_counts=[3; zeros(Int, length(shot.sfe) - 1)],
        Nr=16, Nt=16, sample_count=51,
        outer_its=20, outer_tol=1e-8, tol=1e-9, debug=false)

Run the VEQ direct solve for `shot`'s boundary and pressure/current profiles
and write the result back into `shot`. All Shot profile routes are supported
(dP_dψ or P; F_dF_dψ, Jt, or Jt_R), on :poloidal or :toroidal grids, with
optional `shot.Ip_target` enforced by TEQUILA-style current rescaling.

The flux scale ψ_s = −Ψaxis is converged by an outer fixed point (α2 = ψ_s²),
warm-starting each inner Newton solve. When the profile conversion depends on
the equilibrium (P/Jt/Jt_R, :toroidal grids, or Ip constraint), the solution
is written back into `shot` every outer iteration so conversions and Ip see
the current geometry.

Counts arrays may include trailing zeros: those harmonics stay passive
(boundary value scaled by ρ^K_m) but still shape the geometry.

Returns `shot`.
"""
function veq_solve!(shot::Shot;
    h_count::Int=3, v_count::Int=0, kappa_count::Int=6, c0_count::Int=0, psin_count::Int=6,
    c_counts::AbstractVector{<:Integer}=zeros(Int, length(shot.cfe)),
    s_counts::AbstractVector{<:Integer}=[3; zeros(Int, length(shot.sfe) - 1)],
    Nr::Int=16, Nt::Int=16, sample_count::Int=51,
    outer_its::Int=20, outer_tol::Real=1e-8, tol::Real=1e-9,
    ψ_s0::Real=0.0, debug::Bool=false)

    top = VEQTopology(; h_count, v_count, kappa_count, c0_count, psin_count,
        c_counts, s_counts, Nr, Nt, sample_count)
    boundary = VEQBoundary(shot)
    needs_geometry = veq_needs_geometry(shot)

    # flux scale ψ_s = dΨ/dψn = −Ψaxis (Ψbnd = 0). For an uninitialized Shot
    # (Ψ ≡ 0), start from ψ_s0 (0 = guess from Ip), whose sign selects the
    # flux orientation, and set Ψ to the flux-function form so that
    # equilibrium-dependent profile conversions are defined.
    _, _, Ψaxis = find_axis(shot)
    if Ψaxis == 0.0
        ψ_s = ψ_s0 == 0.0 ? veq_guess_flux_scale(shot) : float(ψ_s0)
        needs_geometry && veq_init_flux!(shot, -ψ_s)
    else
        ψ_s = -Ψaxis
    end

    update_profiles!(shot)
    shot.Ip_target !== nothing && scale_Ip!(shot)
    heat, curr = veq_source_samples(shot, ψ_s, sample_count)
    kern = VEQKernel(top, boundary, VEQSource(; heat_profile=heat, current_profile=curr))

    # Flux-convention note: TEQUILA's Ψ is the total poloidal flux (Wb) with
    # Ψbnd = 0, while VEQ's GS residual is written for per-radian flux
    # ψ = Ψ/2π, so α2 = (dψ/dψn)² = (ψ_s/2π)². The self-consistency condition
    # is therefore α2(ψ_s) = (ψ_s/2π)². For dP_dψ/F_dF_dψ inputs both sources
    # scale linearly with ψ_s, which leaves the solved shape x invariant and
    # makes α2 exactly linear in ψ_s — so the fixed point ψ_s* = 4π²·α2/ψ_s is
    # exact after a single inner solve and the loop finishes in 2-3 iterations.
    # Equilibrium-dependent routes add geometry feedback through the
    # per-iteration writeback and converge like a (fast) Picard iteration.
    x = zeros(kern.x_size)
    local ok, rn, it
    wb_ok = true
    for outer in 1:outer_its
        x, ok, rn, it = veq_solve(kern; x0=x, tol)
        _, α1, α2 = veq_residual!(similar(x), x, kern)
        ψ_s_new = 4π^2 * α2 / ψ_s
        debug && println("outer $outer: ψ_s = $ψ_s → $ψ_s_new, inner its = $it, |r| = $rn")
        # ψ_s stationarity is the convergence measure (like Picard's Ψaxis
        # criterion): geometry, profile conversions, and the Ip rescale factor
        # all feed back into ψ_s, so they are stationary when ψ_s is.
        err = abs(ψ_s_new - ψ_s) / abs(ψ_s_new)
        ψ_s = ψ_s_new
        if needs_geometry
            # profile conversions and Ip must see this iteration's equilibrium;
            # an unconverged state can be non-nested — then keep the previous
            # geometry for this iteration's conversions and let ψ_s/x evolve
            wb_ok = veq_writeback!(shot, x, kern, ψ_s)
            if wb_ok
                update_profiles!(shot)
                shot.Ip_target !== nothing && scale_Ip!(shot)
            else
                debug && println("outer $outer: non-nested surfaces, writeback skipped")
            end
        end
        (err < outer_tol && wb_ok) && break
        heat, curr = veq_source_samples(shot, ψ_s, sample_count)
        kern = veq_with_source(kern, VEQSource(; heat_profile=heat, current_profile=curr))
    end

    ok || @warn "veq_solve!: inner Newton did not reach tol" rn it

    if needs_geometry
        wb_ok || error("veq_solve!: solve ended with non-nested surfaces (inner |r| = $rn); " *
                       "try increasing the VEQ profile counts (psin_count, kappa_count, ...) for this profile shape")
    else
        veq_writeback!(shot, x, kern, ψ_s) ||
            error("veq_solve!: solve produced non-nested surfaces (inner |r| = $rn); " *
                  "try increasing the VEQ profile counts (psin_count, kappa_count, ...) for this profile shape")
    end
    return shot
end

export veq_solve!
