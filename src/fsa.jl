function trapa(f::F1; min_level=3, max_level=20, tol::Real=eps(typeof(1.0))) where {F1}
    int = twopi * f(0.0)
    for l in 1:max_level
        dx = twopi / 2^l
        X = range(0, twopi, 2^l + 1)[2:2:end]
        int2 = dx * sum(f, X)
        int = 0.5 * int + int2
        if l >= min_level
            abs(int <= tol) && break
            (abs(int2 / int - 0.5) <= tol) && break
        end
    end
    return int
end

function Jf_J(x, J, f)
    j = J(x)
    return @SVector[j * f(x), j]
end

function fsa_trapa(J::F1, f::F2; min_level=3, max_level=20, tol::Real=eps(typeof(1.0))) where {F1,F2}
    j = J(0.0)
    int_Jf = twopi * j * f(0.0)
    int_J = twopi * j
    for l in 1:max_level
        dx = twopi / 2^l
        X = range(0, twopi, 2^l + 1)[2:2:end]
        int2_Jf, int2_J = dx .* sum(x -> Jf_J(x, J, f), X)
        int_Jf = 0.5 * int_Jf + int2_Jf
        int_J = 0.5 * int_J + int2_J
        if l >= min_level
            abs(int_Jf <= tol) && break
            (abs(int2_Jf / int_Jf - 0.5) <= tol) && (abs(int2_J / int_J - 0.5) <= tol) && break
        end
    end

    return int_Jf / int_J
end

"""
    Vprime(shot::F1, ρ::Real) where {F1<:Shot}

Compute dV/dρ at `ρ` for the equilibrium defined in `shot`
"""
function Vprime(shot::F1, ρ::Real) where {F1<:Shot}
    k, nu_ou, nu_eu, nu_ol, nu_el, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el = compute_both_bases(shot.ρ, ρ)
    R0x = evaluate_inbounds(shot.R0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ϵx = evaluate_inbounds(shot.ϵfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    κx = evaluate_inbounds(shot.κfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    c0x = evaluate_inbounds(shot.c0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    cx, sx = evaluate_csx!(shot, k, nu_ou, nu_eu, nu_ol, nu_el)

    dR0x = evaluate_inbounds(shot.R0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dZ0x = evaluate_inbounds(shot.Z0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dϵx = evaluate_inbounds(shot.ϵfe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dκx = evaluate_inbounds(shot.κfe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dc0x = evaluate_inbounds(shot.c0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dcx, dsx = evaluate_dcsx!(shot, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)

    J = θ -> MillerExtendedHarmonic.Jacobian(θ, R0x, ϵx, κx, c0x, cx, sx, dR0x, dZ0x, dϵx, dκx, dc0x, dcx, dsx)
    return twopi * trapa(J)
end

"""
    FSA(f::F1, shot::F2, ρ::Real) where {F1,F2<:Shot}

Compute flux-surface average of `f` at `ρ` for the equilibrium defined in `shot`

Here `f` is a function of `θ` only, so something like f = θ -> F(ρ, θ) may be required
"""
function FSA(f::F1, shot::F2, ρ::Real) where {F1,F2<:Shot}
    ρ == 0.0 && return f(0.0)

    k, nu_ou, nu_eu, nu_ol, nu_el, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el = compute_both_bases(shot.ρ, ρ)
    R0x = evaluate_inbounds(shot.R0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ϵx = evaluate_inbounds(shot.ϵfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    κx = evaluate_inbounds(shot.κfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    c0x = evaluate_inbounds(shot.c0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    cx, sx = evaluate_csx!(shot, k, nu_ou, nu_eu, nu_ol, nu_el)

    dR0x = evaluate_inbounds(shot.R0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dZ0x = evaluate_inbounds(shot.Z0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dϵx = evaluate_inbounds(shot.ϵfe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dκx = evaluate_inbounds(shot.κfe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dc0x = evaluate_inbounds(shot.c0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dcx, dsx = evaluate_dcsx!(shot, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)

    J = θ -> MillerExtendedHarmonic.Jacobian(θ, R0x, ϵx, κx, c0x, cx, sx, dR0x, dZ0x, dϵx, dκx, dc0x, dcx, dsx)
    return fsa_trapa(J, f)
end

"""
    FSA(f::F1, shot::F2, ρ::Real, Vprime::F3) where {F1,F2<:Shot,F3<:FE_rep}

Compute flux-surface average of `f` at `ρ` for the equilibrium defined in `shot`,
using the given finite-element representation of `Vprime`

Here `f` is a function of `θ` only, so something like f = θ -> F(ρ, θ) may be required
"""
function FSA(f::F1, shot::F2, ρ::Real, Vprime::F3) where {F1,F2<:Shot,F3<:FE_rep}
    ρ == 0.0 && return f(0.0)

    k, nu_ou, nu_eu, nu_ol, nu_el, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el = compute_both_bases(shot.ρ, ρ)
    R0x = evaluate_inbounds(shot.R0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ϵx = evaluate_inbounds(shot.ϵfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    κx = evaluate_inbounds(shot.κfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    c0x = evaluate_inbounds(shot.c0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    cx, sx = evaluate_csx!(shot, k, nu_ou, nu_eu, nu_ol, nu_el)

    dR0x = evaluate_inbounds(shot.R0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dZ0x = evaluate_inbounds(shot.Z0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dϵx = evaluate_inbounds(shot.ϵfe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dκx = evaluate_inbounds(shot.κfe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dc0x = evaluate_inbounds(shot.c0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dcx, dsx = evaluate_dcsx!(shot, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)

    Vp = evaluate_inbounds(Vprime, k, nu_ou, nu_eu, nu_ol, nu_el)
    Jf = θ -> f(θ) * MillerExtendedHarmonic.Jacobian(θ, R0x, ϵx, κx, c0x, cx, sx, dR0x, dZ0x, dϵx, dκx, dc0x, dcx, dsx)

    return twopi * trapa(Jf) / Vp
end

"""
    FSA(f::F1, shot::F2, ρ::Real, Vprime::Real) where {F1,F2<:Shot}

Compute flux-surface average of `f` at `ρ` for the equilibrium defined in `shot`,
using the given value of `Vprime`

Here `f` is a function of `θ` only, so something like f = θ -> F(ρ, θ) may be required
"""
function FSA(f::F1, shot::F2, ρ::Real, Vprime::Real) where {F1,F2<:Shot}
    ρ == 0.0 && return f(0.0)

    k, nu_ou, nu_eu, nu_ol, nu_el, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el = compute_both_bases(shot.ρ, ρ)
    R0x = evaluate_inbounds(shot.R0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ϵx = evaluate_inbounds(shot.ϵfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    κx = evaluate_inbounds(shot.κfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    c0x = evaluate_inbounds(shot.c0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    cx, sx = evaluate_csx!(shot, k, nu_ou, nu_eu, nu_ol, nu_el)

    dR0x = evaluate_inbounds(shot.R0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dZ0x = evaluate_inbounds(shot.Z0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dϵx = evaluate_inbounds(shot.ϵfe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dκx = evaluate_inbounds(shot.κfe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dc0x = evaluate_inbounds(shot.c0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dcx, dsx = evaluate_dcsx!(shot, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)

    Jf = θ -> f(θ) * MillerExtendedHarmonic.Jacobian(θ, R0x, ϵx, κx, c0x, cx, sx, dR0x, dZ0x, dϵx, dκx, dc0x, dcx, dsx)
    return twopi * trapa(Jf) / Vprime
end

"""
    fsa_invR2(shot::F1, ρ) where {F1<:Shot}

Compute <R⁻²> at `ρ` for the equilibrium defined in `shot`
"""
function fsa_invR2(shot::F1, ρ) where {F1<:Shot}
    k, nu_ou, nu_eu, nu_ol, nu_el = compute_bases(shot.ρ, ρ)
    R0x = evaluate_inbounds(shot.R0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ϵx = evaluate_inbounds(shot.ϵfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    c0x = evaluate_inbounds(shot.c0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    cx, sx = evaluate_csx!(shot, k, nu_ou, nu_eu, nu_ol, nu_el)
    ax = R0x * ϵx

    f = θ -> MillerExtendedHarmonic.R_MXH(θ, R0x, c0x, cx, sx, ax)^-2

    return FSA(f, shot, ρ)
end

"""
    fsa_invR(shot::F1, ρ) where {F1<:Shot}

Compute <R⁻¹> at `ρ` for the equilibrium defined in `shot`
"""
function fsa_invR(shot::F1, ρ) where {F1<:Shot}
    k, nu_ou, nu_eu, nu_ol, nu_el = compute_bases(shot.ρ, ρ)
    R0x = evaluate_inbounds(shot.R0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ϵx = evaluate_inbounds(shot.ϵfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    c0x = evaluate_inbounds(shot.c0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    cx, sx = evaluate_csx!(shot, k, nu_ou, nu_eu, nu_ol, nu_el)
    ax = R0x * ϵx

    f = θ -> MillerExtendedHarmonic.R_MXH(θ, R0x, c0x, cx, sx, ax)^-1

    return FSA(f, shot, ρ)
end

function FiniteElementHermite.FE_rep(shot::F1, f::F2, coeffs=Vector{typeof(shot.ρ[1])}(undef, 2 * length(shot.ρ)); ε=1e-6) where {F1<:Shot,F2}
    for (i, x) in enumerate(shot.ρ)
        # BCL 4/26/23: I'd like to use ForwardDiff here, but the use of intermediate arrays
        #              in evaluate_csx!() prevents that
        # BCL 6/2/23: Can use ForwardDiff now but it gives slightly different results,
        #             maybe due to issues at boundary? Don't know what is best but
        #             going with ForwardDiff
        g = x -> f(shot, x)
        if x == 0.0
            coeffs[2i-1] = ForwardDiff.derivative(g, 1e-12)
        else
            coeffs[2i-1] = ForwardDiff.derivative(g, x)
        end
        coeffs[2i] = g(x)
    end
    return FE_rep(shot.ρ, coeffs)
end

function FE_coeffs!(Y::FE_rep, shot::F1, f::F2; ε::Real=1e-6, derivative::Symbol=:auto) where {F1<:Shot,F2}
    @assert derivative in (:auto, :finite)
    (derivative === :auto) && (g = x -> f(shot, x))
    for (i, x) in enumerate(Y.x)
        Y.coeffs[2i] = f(shot, x)
        if derivative === :auto
            if x == 0.0
                Y.coeffs[2i-1] = ForwardDiff.derivative(g, 1e-12)
            else
                Y.coeffs[2i-1] = ForwardDiff.derivative(g, x)
            end
        else
            if x == Y.x[end]
                xp = x
                fp = Y.coeffs[2i]
            else
                xp =  x + ε * (Y.x[i+1] - Y.x[i])
                fp = f(shot, xp)
            end

            if x == Y.x[1]
                xm = x
                fm = Y.coeffs[2i]
            else
                xm = x - ε * (Y.x[i] - Y.x[i-1])
                fm = f(shot, xm)
            end

            Y.coeffs[2i-1] = (fp - fm) / (xp - xm)
        end


    end
    return Y
end

function Fpol_coeffs!(Y::FE_rep, shot::F1; invR=FE_rep(shot, fsa_invR), invR2=FE_rep(shot, fsa_invR2)) where {F1<:Shot}
    F = shot.Fbnd
    F_dF_dψ= FFprime(shot, shot.F_dF_dψ, shot.Jt_R, shot.Jt; invR, invR2)
    FFp = x -> F_dF_dψ(x) * dψ_dρ(shot, x)
    for i in reverse(eachindex(Y.x))
        ρ = Y.x[i]
        if i < length(Y.x)
            # just integrate from ρ to Y.x[i+1],
            #   and use previous F as boundary condition
            endpoint = (Y.x[i+1], F)
            F = Fpol(shot, ρ, endpoint; invR, invR2)
        end
        Y.coeffs[2i] = F
        Y.coeffs[2i-1] = FFp(ρ) / F
    end
    return Y
end

function _int_J_invR2(shot::F1, ρ::Real) where {F1<:Shot}
    k, nu_ou, nu_eu, nu_ol, nu_el, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el = compute_both_bases(shot.ρ, ρ)
    R0x = evaluate_inbounds(shot.R0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ϵx = evaluate_inbounds(shot.ϵfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    κx = evaluate_inbounds(shot.κfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    c0x = evaluate_inbounds(shot.c0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    cx, sx = evaluate_csx!(shot, k, nu_ou, nu_eu, nu_ol, nu_el)
    ax = R0x * ϵx

    dR0x = evaluate_inbounds(shot.R0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dZ0x = evaluate_inbounds(shot.Z0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dϵx = evaluate_inbounds(shot.ϵfe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dκx = evaluate_inbounds(shot.κfe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dc0x = evaluate_inbounds(shot.c0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dcx, dsx = evaluate_dcsx!(shot, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)

    J_invR2 = θ -> (MillerExtendedHarmonic.Jacobian(θ, R0x, ϵx, κx, c0x, cx, sx, dR0x, dZ0x, dϵx, dκx, dc0x, dcx, dsx) * MillerExtendedHarmonic.R_MXH(θ, R0x, c0x, cx, sx, ax)^-2)

    return trapa(J_invR2)
end

function toroidal_flux(shot::F1, ρ::Real) where {F1<:Shot}
    invR2 = FE_rep(shot, fsa_invR2)
    Vp = FE_rep(shot, Vprime)
    invR = FE_rep(shot, fsa_invR)
    F = x -> Fpol(shot, x; invR, invR2)
    return toroidal_flux(ρ, F, Vp, invR2)
end

function toroidal_flux(ρ::Real, F, Vp, invR2)
    f = x -> F(x) * Vp(x) * invR2(x)
    return quadgk(f, 0.0, ρ)[1] / twopi
end

function toroidal_flux(shot::F1, ρs::AbstractVector{<:Real}) where {F1<:Shot}
    Φ = zero(ρs)
    return toroidal_flux!(Φ, shot, ρs)
end

function toroidal_flux!(Φ::AbstractVector{<:Real}, shot::F1, ρs::AbstractVector{<:Real}; use_cached=true) where {F1<:Shot}
    @assert length(Φ) === length(ρs)
    Vp, _, invR2, F = get_FEs(shot, use_cached)
    f = x -> F(x) * Vp(x) * invR2(x)
    for k in eachindex(ρs)[2:end]
        Φ[k] = Φ[k-1] + quadgk(f, ρs[k-1], ρs[k])[1] / twopi
    end
    return Φ
end

function rho_tor_norm(shot::F1, ρ::Real) where {F1<:Shot}
    Φ = toroidal_flux(ρ, shot.F, shot.Vp, shot.invR2)
    Φ0 = toroidal_flux(1.0, shot.F, shot.Vp, shot.invR2)
    return sqrt(abs(Φ / Φ0)) # abs prevents -0.0
end

function rho_tor_norm(shot::F1) where {F1<:Shot}
    ρtor = zero(shot.ρ)
    return rho_tor_norm!(ρtor, shot)
end

function rho_tor_norm!(ρtor::AbstractVector{<:Real}, shot::F1) where {F1<:Shot}
    @assert length(ρtor) === length(shot.ρ)
    toroidal_flux!(ρtor, shot, shot.ρ)
    @. ρtor = sqrt(ρtor / ρtor[end])
    ρtor[1] = 0.0
    ρtor[end] = 1.0
    return ρtor
end

function get_FEs(shot::Shot, use_cached::Bool)
    Vp    = use_cached ? shot.Vp : FE_rep(shot, Vprime)
    invR  = use_cached ? shot.invR : FE_rep(shot, fsa_invR)
    invR2 = use_cached ? shot.invR2 : FE_rep(shot, fsa_invR2)
    if use_cached
        F = shot.F
    else
        N = length(shot.ρ)
        F = FE_rep(shot.ρ, zeros(eltype(shot.ρ, 2N)))
        Fpol_coeffs!(F, shot; invR, invR2)
    end

    return Vp, invR, invR2, F
end

function ρtor_coeffs!(Y::FE_rep, shot::F1; use_cached=true, ε::Real=1e-6) where {F1<:Shot}
    @assert length(Y.x) === length(shot.ρ)

    Vp, _, invR2, F = get_FEs(shot, use_cached)

    # to start, compute Φ
    @views Φ = Y.coeffs[2:2:end]
    toroidal_flux!(Φ, shot, shot.ρ; use_cached)
    Φ0 = Φ[end]

    # then compute ρtor
    @. Y.coeffs[2:2:end] = sqrt(Φ / Φ0)
    Y.coeffs[1] = 0.0
    Y.coeffs[end] = 1.0
    @views ρtor = Y.coeffs[2:2:end]

    # compute derivative analytically, since Φ is an integral
    # dρtor_dρ = dΦ_dρ / (2 * Φ0 * ρtor)
    f = x -> F(x) * Vp(x) * invR2(x) / twopi
    @views ρtor = Y.coeffs[2:2:end]
    Y.coeffs[1:2:end] .= f.(Y.x) ./ (2.0 .* Φ0 .* ρtor)

    # fix on-axis derivative by taking approximate derivative at δ
    # using ρtor(δ) ≈ dρtor_dρ(δ) * δ  in equation above
    δ = ε * Y.x[2]
    Y.coeffs[1] = sqrt(f(δ) / (2.0 * Φ0 * δ))

    return Y
end

function set_FSAs!(shot)
    FE_coeffs!(shot.Vp, shot, Vprime; derivative=:auto)
    FE_coeffs!(shot.invR, shot, fsa_invR; derivative=:auto)
    FE_coeffs!(shot.invR2, shot, fsa_invR2; derivative=:auto)
    Fpol_coeffs!(shot.F, shot; shot.invR, shot.invR2)
    ρtor_coeffs!(shot.ρtor, shot)
    return shot
end

function Ψ(shot)
    @views Psi = FE_rep(shot.ρ, shot.C[:, 1])
    return Psi
end

"""
    Ip(shot::F1; ε::Real=1e-6) where {F1<:Shot}

Returns the plasma current of `shot`
"""
function Ip(shot::F1) where {F1<:Shot}
    Vp = FE_rep(shot, Vprime)
    if shot.Jt_R !== nothing
        f1 = x -> Vp(x) * shot.Jt_R(x)
        return quadgk(f1, 0.0, 1.0)[1] / twopi
    elseif shot.Jt !== nothing
        invR = FE_rep(shot, fsa_invR)
        f2 = x -> Vp(x) * shot.Jt(x) * invR(x)
        return quadgk(f2, 0.0, 1.0)[1] / twopi
    else
        invR2 = FE_rep(shot, fsa_invR2)
        Pp = Pprime(shot, shot.P, shot.dP_dψ)
        f3 = x -> -Vp(x) * (Pp(x) + invR2(x) * shot.F_dF_dψ(x) / μ₀)
        return quadgk(f3, 0.0, 1.0)[1]
    end
    return 0.0
end

function Ip_ffp(shot::F1) where {F1<:Shot}
    (shot.F_dF_dψ === nothing) && return 0.0

    Vp = FE_rep(shot, Vprime)
    invR2 = FE_rep(shot, fsa_invR2)
    f = x -> -Vp(x) * invR2(x) * shot.F_dF_dψ(x) / μ₀
    return quadgk(f, 0.0, 1.0)[1]
end
