#= Forward-mode sensitivities of the VEQ solve.

`veq_solve!` on a Shot whose inputs (profile coefficients, Pbnd, Fbnd,
Ip_target) are ForwardDiff duals does not iterate in duals. It

 1. strips the partials and runs the unchanged Float64 solve (`veq_solve_values!`),
    so the values are those of a plain Float64 run;
 2. linearizes the converged outer map about that solution: each sensitivity
    pass rebuilds the dual sources from the dual shot (profile conversion, Ip
    rescale), takes one Newton step of the packed residual with the converged
    Float64 Jacobian (exact for the partials: the implicit function theorem),
    updates the flux scale ψ_s = 4π²·α2/ψ_s and writes the dual state back into
    the shot. Values are reset to the Float64 solution after every pass, so
    only the partials iterate; they converge at the rate of the outer
    fixed point (a few passes).

The returned Shot is a new object with dual number type (callers must use the
return value, as for `solve`).
=#

_value(x::ForwardDiff.Dual) = ForwardDiff.value(x)
_value(x::Real) = x

_fe_numtype(fe::FE_rep) = eltype(fe.coeffs)
_fe_numtype(::Function) = Float64

"""
    veq_input_numtype(shot::Shot)

Promoted number type of the Shot's geometry and solve inputs (profiles,
Pbnd, Fbnd, Ip_target). Profiles given as plain functions count as Float64.
"""
function veq_input_numtype(shot::Shot)
    T = promote_type(eltype(shot.surfaces), typeof(shot.Pbnd), typeof(shot.Fbnd))
    shot.Ip_target === nothing || (T = promote_type(T, typeof(shot.Ip_target)))
    for prof in (shot.P, shot.dP_dψ, shot.F_dF_dψ, shot.Jt_R, shot.Jt)
        prof === nothing && continue
        T = promote_type(T, eltype(prof.fe.coeffs), _fe_numtype(prof.orig))
    end
    return T
end

_retype_orig(orig::FE_rep, f) = _retype_fe(orig, f)
_retype_orig(orig::Function, f) = orig
_retype_profile(::Nothing, f) = nothing
_retype_profile(prof::Profile, f) = Profile(_retype_fe(prof.fe, f), _retype_orig(prof.orig, f), prof.grid)

"""
    _retype_shot(geom::Shot, inputs::Shot, fg, fi)

New Shot with the geometry (surfaces, Ψ, shape FEs) of `geom` mapped through
`fg` and the profiles and scalars of `inputs` mapped through `fi`. The shape
FEs are mapped, not refitted, so edge-derivative corrections are kept; the
quadrature and flux-surface averages are recomputed from them.
"""
function _retype_shot(geom::Shot, inputs::Shot, fg, fi)
    surfaces = map(fg, geom.surfaces)
    C = map(fg, geom.C)
    R0fe, Z0fe, ϵfe, κfe, c0fe = (_retype_fe(fe, fg) for fe in (geom.R0fe, geom.Z0fe, geom.ϵfe, geom.κfe, geom.c0fe))
    cfe = [_retype_fe(fe, fg) for fe in geom.cfe]
    sfe = [_retype_fe(fe, fg) for fe in geom.sfe]
    Q = QuadInfo(geom.ρ, geom.M, length(cfe), R0fe, Z0fe, ϵfe, κfe, c0fe, cfe, sfe)
    Ip_target = inputs.Ip_target === nothing ? nothing : fi(inputs.Ip_target)
    return Shot(geom.N, geom.M, geom.ρ, surfaces, C, R0fe, Z0fe, ϵfe, κfe, c0fe, cfe, sfe, Q;
        P=_retype_profile(inputs.P, fi), dP_dψ=_retype_profile(inputs.dP_dψ, fi),
        F_dF_dψ=_retype_profile(inputs.F_dF_dψ, fi), Jt_R=_retype_profile(inputs.Jt_R, fi),
        Jt=_retype_profile(inputs.Jt, fi), Pbnd=fi(inputs.Pbnd), Fbnd=fi(inputs.Fbnd), Ip_target)
end

"""Plain-float copy of `shot` (all partials dropped)."""
value_shot(shot::Shot) = _retype_shot(shot, shot, _value, _value)

# value `v` with the partials of `d`
_with_value(v::Real, d::ForwardDiff.Dual{T}) where {T} = ForwardDiff.Dual{T}(v, ForwardDiff.partials(d))

"""
    _dual_solve(F::Factorization, r)

`F \\ r` for a right-hand side of duals and a plain-float factorization:
values and each partial direction are solved separately.
"""
_dual_solve(F::Factorization, r::AbstractVector{<:AbstractFloat}) = F \ r
function _dual_solve(F::Factorization, r::AbstractVector{ForwardDiff.Dual{T,V,N}}) where {T,V,N}
    v = _dual_solve(F, ForwardDiff.value.(r))
    P = [_dual_solve(F, [ForwardDiff.partials(ri, j) for ri in r]) for j in 1:N]
    return [ForwardDiff.Dual{T}(v[i], ForwardDiff.Partials(ntuple(j -> P[j][i], Val(N)))) for i in eachindex(v)]
end

_partials_vec(d::ForwardDiff.Dual) = collect(ForwardDiff.partials(d))
_partials_change(a::ForwardDiff.Dual, b::ForwardDiff.Dual) =
    maximum(abs, _partials_vec(a) .- _partials_vec(b); init=0.0) / max(maximum(abs, _partials_vec(a); init=0.0), floatmin())
function _partials_change(a::AbstractVector, b::AbstractVector)
    Pa = reduce(hcat, _partials_vec.(a))
    Pb = reduce(hcat, _partials_vec.(b))
    return maximum(abs, Pa .- Pb; init=0.0) / max(maximum(abs, Pa; init=0.0), floatmin())
end

# kernel on the converged topology/grid with a new boundary and source
function _veq_kernel(kern::VEQKernel, boundary::VEQBoundary, source::VEQSource)
    return VEQKernel(kern.top, kern.grid, kern.blocks, boundary, source,
        veq_spline_coefficients(μ₀ .* source.heat_profile),
        veq_spline_coefficients(source.current_profile),
        kern.x_size, Dict{DataType,Any}())
end

"""
    veq_solve_dual(shot::Shot; sens_its=20, sens_tol=1e-10, kwargs...)

Dual-number path of `veq_solve!` (see the header of this file). `kwargs` are
those of `veq_solve!`. Returns a new Shot of the inputs' dual number type.
"""
function veq_solve_dual(shot::Shot; sens_its::Int=20, sens_tol::Real=1e-7, debug::Bool=false, kwargs...)
    D = veq_input_numtype(shot)
    ForwardDiff.valtype(D) === Float64 ||
        throw(ArgumentError("veq_solve!: nested dual inputs ($D) are not supported"))
    sample_count = get(kwargs, :sample_count, 51)

    # 1. Float64 solve
    shot_v, x_v, kern_v, ψs_v, _ = veq_solve_values!(value_shot(shot); debug, kwargs...)
    needs_geometry = veq_needs_geometry(shot_v)

    # Jacobian of the packed residual at the converged state (values only)
    f!(out, xx) = (veq_residual!(out, xx, kern_v); out)
    J = ForwardDiff.jacobian(f!, similar(x_v), x_v)
    Jfac = lu(J; check=false)
    issuccess(Jfac) || error("veq_solve!: singular VEQ Jacobian at the converged state; cannot propagate derivatives")

    # 2. dual shot: converged geometry (no partials yet) + the dual inputs
    toD = y -> convert(D, y)
    shot_d = _retype_shot(shot_v, shot, toD, toD)
    boundary_d = VEQBoundary(shot_d)
    x = toD.(x_v)
    ψs = toD(ψs_v)

    Δ = Inf
    for pass in 1:sens_its
        if needs_geometry
            update_profiles!(shot_d)
            shot_d.Ip_target !== nothing && scale_Ip!(shot_d)
        end
        heat, curr = veq_source_samples(shot_d, ψs, sample_count)
        kern = _veq_kernel(kern_v, boundary_d, VEQSource(; heat_profile=heat, current_profile=curr))
        r = veq_residual(x, kern)
        xn = _with_value.(x_v, x .- _dual_solve(Jfac, r))
        _, _, α2 = veq_residual!(similar(r), xn, kern)
        ψsn = _with_value(ψs_v, 4π^2 * α2 / ψs)
        Δ = max(_partials_change(xn, x), _partials_change(ψsn, ψs))
        x, ψs = xn, ψsn
        debug && println("veq sensitivity pass $pass: max relative partials change = $Δ")
        if needs_geometry || Δ <= sens_tol || pass == sens_its
            veq_writeback!(shot_d, x, kern, ψs) ||
                error("veq_solve!: dual writeback produced non-nested surfaces")
        end
        Δ <= sens_tol && break
    end
    Δ <= sens_tol || @warn "veq_solve!: derivative passes did not converge" Δ sens_its
    if needs_geometry
        update_profiles!(shot_d)
        shot_d.Ip_target !== nothing && scale_Ip!(shot_d)
    end
    return shot_d
end
