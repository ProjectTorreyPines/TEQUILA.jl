function trapa(f; min_level=3, max_level=20, tol::Real=eps(typeof(1.0)))
    int = twopi * f(0.0)
    for l in 1:max_level
        dx = twopi / 2^l
        X = range(0, twopi, 2^l + 1)[2:2:end]
        int2 = dx * sum(f(x) for x in X)
        int = 0.5 * int + int2
        if l >= min_level
            abs(int <= tol) && break
            (abs(int2 / int - 0.5) <= tol) && break
        end
    end
    return int
end

function Vprime(shot::Shot, ρ::Real)
    k, nu_ou, nu_eu, nu_ol, nu_el, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el = compute_both_bases(shot.ρ, ρ)
    R0x = evaluate_inbounds(shot.R0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ϵx = evaluate_inbounds(shot.ϵfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    κx = evaluate_inbounds(shot.κfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    c0x = evaluate_inbounds(shot.c0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    evaluate_csx!(shot, k, nu_ou, nu_eu, nu_ol, nu_el)

    dR0x = evaluate_inbounds(shot.R0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dZ0x = evaluate_inbounds(shot.Z0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dϵx = evaluate_inbounds(shot.ϵfe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dκx = evaluate_inbounds(shot.κfe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dc0x = evaluate_inbounds(shot.c0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    evaluate_dcsx!(shot, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)

    J(θ) = MillerExtendedHarmonic.Jacobian(θ, R0x, ϵx, κx, c0x, shot._cx, shot._sx, dR0x, dZ0x, dϵx, dκx, dc0x, shot._dcx, shot._dsx)
    return twopi * trapa(J)
end

function FSA(f, shot::Shot, ρ::Real)

    ρ == 0.0 && return f(0.0)

    k, nu_ou, nu_eu, nu_ol, nu_el, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el = compute_both_bases(shot.ρ, ρ)
    R0x = evaluate_inbounds(shot.R0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ϵx = evaluate_inbounds(shot.ϵfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    κx = evaluate_inbounds(shot.κfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    c0x = evaluate_inbounds(shot.c0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    evaluate_csx!(shot, k, nu_ou, nu_eu, nu_ol, nu_el)

    dR0x = evaluate_inbounds(shot.R0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dZ0x = evaluate_inbounds(shot.Z0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dϵx = evaluate_inbounds(shot.ϵfe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dκx = evaluate_inbounds(shot.κfe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dc0x = evaluate_inbounds(shot.c0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    evaluate_dcsx!(shot, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)

    J(θ) = MillerExtendedHarmonic.Jacobian(θ, R0x, ϵx, κx, c0x, shot._cx, shot._sx, dR0x, dZ0x, dϵx, dκx, dc0x, shot._dcx, shot._dsx)
    Jf(θ) = f(θ) * J(θ)
    return trapa(Jf) / trapa(J)
end

function FSA(f, shot::Shot, ρ::Real, Vprime::FE_rep)

    ρ == 0.0 && return f(0.0)

    k, nu_ou, nu_eu, nu_ol, nu_el, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el = compute_both_bases(shot.ρ, ρ)
    R0x = evaluate_inbounds(shot.R0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ϵx = evaluate_inbounds(shot.ϵfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    κx = evaluate_inbounds(shot.κfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    c0x = evaluate_inbounds(shot.c0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    evaluate_csx!(shot, k, nu_ou, nu_eu, nu_ol, nu_el)

    dR0x = evaluate_inbounds(shot.R0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dZ0x = evaluate_inbounds(shot.Z0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dϵx = evaluate_inbounds(shot.ϵfe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dκx = evaluate_inbounds(shot.κfe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dc0x = evaluate_inbounds(shot.c0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    evaluate_dcsx!(shot, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)

    Vp = evaluate_inbounds(Vprime, k, nu_ou, nu_eu, nu_ol, nu_el)
    Jf(θ) = f(θ) * MillerExtendedHarmonic.Jacobian(θ, R0x, ϵx, κx, c0x, shot._cx, shot._sx, dR0x, dZ0x, dϵx, dκx, dc0x, shot._dcx, shot._dsx)

    return twopi * trapa(Jf) / Vp
end

function FSA(f, shot::Shot, ρ::Real, Vprime::Real)

    ρ == 0.0 && return f(0.0)

    k, nu_ou, nu_eu, nu_ol, nu_el, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el = compute_both_bases(shot.ρ, ρ)
    R0x = evaluate_inbounds(shot.R0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ϵx = evaluate_inbounds(shot.ϵfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    κx = evaluate_inbounds(shot.κfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    c0x = evaluate_inbounds(shot.c0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    evaluate_csx!(shot, k, nu_ou, nu_eu, nu_ol, nu_el)

    dR0x = evaluate_inbounds(shot.R0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dZ0x = evaluate_inbounds(shot.Z0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dϵx = evaluate_inbounds(shot.ϵfe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dκx = evaluate_inbounds(shot.κfe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dc0x = evaluate_inbounds(shot.c0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    evaluate_dcsx!(shot, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)

    Jf(θ) = f(θ) * MillerExtendedHarmonic.Jacobian(θ, R0x, ϵx, κx, c0x, shot._cx, shot._sx, dR0x, dZ0x, dϵx, dκx, dc0x, shot._dcx, shot._dsx)
    return twopi * trapa(Jf) / Vprime
end

function fsa_invR2(shot, ρ)
    k, nu_ou, nu_eu, nu_ol, nu_el = compute_bases(shot.ρ, ρ)
    R0x = evaluate_inbounds(shot.R0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ϵx = evaluate_inbounds(shot.ϵfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    c0x = evaluate_inbounds(shot.c0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    evaluate_csx!(shot, k, nu_ou, nu_eu, nu_ol, nu_el)
    ax = R0x * ϵx

    f(θ) = MillerExtendedHarmonic.R_MXH(θ, R0x, c0x, shot._cx, shot._sx, ax) ^ -2

    return FSA(f, shot, ρ)
end

function fsa_invR(shot, ρ)
    k, nu_ou, nu_eu, nu_ol, nu_el = compute_bases(shot.ρ, ρ)
    R0x = evaluate_inbounds(shot.R0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ϵx = evaluate_inbounds(shot.ϵfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    c0x = evaluate_inbounds(shot.c0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    evaluate_csx!(shot, k, nu_ou, nu_eu, nu_ol, nu_el)
    ax = R0x * ϵx

    f(θ) = MillerExtendedHarmonic.R_MXH(θ, R0x, c0x, shot._cx, shot._sx, ax) ^ -1

    return FSA(f, shot, ρ)
end

function FE_fsa(shot, fsa, coeffs = Vector{typeof(shot.ρ[1])}(undef, 2*length(shot.ρ)); ε = 1e-6)
    for (i, x) in enumerate(shot.ρ)
        # BCL 4/26/23: I'd like to use ForwardDiff here, but the use of intermediate arrays
        #              in evaluate_csx!() prevents that
        xp = x==shot.ρ[end] ? 1.0 : x + ε * (shot.ρ[i+1] - shot.ρ[i])
        xm = x==shot.ρ[1]   ? 0.0 : x - ε * (shot.ρ[i] - shot.ρ[i-1])
        coeffs[2i-1] = (fsa(shot, xp) - fsa(shot, xm)) / (xp - xm)
        coeffs[2i] = fsa(shot, x)
    end
    return FE_rep(shot.ρ, coeffs)
end

function Ψ(shot)
    @views Psi = FE_rep(shot.ρ, shot.C[:,1])
    return Psi
end