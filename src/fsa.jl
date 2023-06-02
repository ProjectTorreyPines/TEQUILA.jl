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

function Vprime(shot::Shot, ρ::Real; tid = Threads.threadid())
    k, nu_ou, nu_eu, nu_ol, nu_el, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el = compute_both_bases(shot.ρ, ρ)
    R0x = evaluate_inbounds(shot.R0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ϵx = evaluate_inbounds(shot.ϵfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    κx = evaluate_inbounds(shot.κfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    c0x = evaluate_inbounds(shot.c0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    evaluate_csx!(shot, k, nu_ou, nu_eu, nu_ol, nu_el; tid)

    dR0x = evaluate_inbounds(shot.R0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dZ0x = evaluate_inbounds(shot.Z0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dϵx = evaluate_inbounds(shot.ϵfe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dκx = evaluate_inbounds(shot.κfe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dc0x = evaluate_inbounds(shot.c0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    evaluate_dcsx!(shot, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el; tid)

    J(θ) = MillerExtendedHarmonic.Jacobian(θ, R0x, ϵx, κx, c0x, shot._cx[tid], shot._sx[tid], dR0x, dZ0x, dϵx, dκx, dc0x, shot._dcx[tid], shot._dsx[tid])
    return twopi * trapa(J)
end

function FSA(f, shot::Shot, ρ::Real; tid = Threads.threadid())

    ρ == 0.0 && return f(0.0)

    k, nu_ou, nu_eu, nu_ol, nu_el, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el = compute_both_bases(shot.ρ, ρ)
    R0x = evaluate_inbounds(shot.R0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ϵx = evaluate_inbounds(shot.ϵfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    κx = evaluate_inbounds(shot.κfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    c0x = evaluate_inbounds(shot.c0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    evaluate_csx!(shot, k, nu_ou, nu_eu, nu_ol, nu_el; tid)

    dR0x = evaluate_inbounds(shot.R0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dZ0x = evaluate_inbounds(shot.Z0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dϵx = evaluate_inbounds(shot.ϵfe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dκx = evaluate_inbounds(shot.κfe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dc0x = evaluate_inbounds(shot.c0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    evaluate_dcsx!(shot, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el; tid)

    J(θ) = MillerExtendedHarmonic.Jacobian(θ, R0x, ϵx, κx, c0x, shot._cx[tid], shot._sx[tid], dR0x, dZ0x, dϵx, dκx, dc0x, shot._dcx[tid], shot._dsx[tid])
    Jf(θ) = f(θ) * J(θ)
    return trapa(Jf) / trapa(J)
end

function FSA(f, shot::Shot, ρ::Real, Vprime::FE_rep; tid = Threads.threadid())

    ρ == 0.0 && return f(0.0)

    k, nu_ou, nu_eu, nu_ol, nu_el, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el = compute_both_bases(shot.ρ, ρ)
    R0x = evaluate_inbounds(shot.R0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ϵx = evaluate_inbounds(shot.ϵfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    κx = evaluate_inbounds(shot.κfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    c0x = evaluate_inbounds(shot.c0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    evaluate_csx!(shot, k, nu_ou, nu_eu, nu_ol, nu_el; tid)

    dR0x = evaluate_inbounds(shot.R0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dZ0x = evaluate_inbounds(shot.Z0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dϵx = evaluate_inbounds(shot.ϵfe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dκx = evaluate_inbounds(shot.κfe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dc0x = evaluate_inbounds(shot.c0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    evaluate_dcsx!(shot, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el; tid)

    Vp = evaluate_inbounds(Vprime, k, nu_ou, nu_eu, nu_ol, nu_el)
    Jf(θ) = f(θ) * MillerExtendedHarmonic.Jacobian(θ, R0x, ϵx, κx, c0x, shot._cx[tid], shot._sx[tid], dR0x, dZ0x, dϵx, dκx, dc0x, shot._dcx[tid], shot._dsx[tid])

    return twopi * trapa(Jf) / Vp
end

function FSA(f, shot::Shot, ρ::Real, Vprime::Real; tid = Threads.threadid())

    ρ == 0.0 && return f(0.0)

    k, nu_ou, nu_eu, nu_ol, nu_el, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el = compute_both_bases(shot.ρ, ρ)
    R0x = evaluate_inbounds(shot.R0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ϵx = evaluate_inbounds(shot.ϵfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    κx = evaluate_inbounds(shot.κfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    c0x = evaluate_inbounds(shot.c0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    evaluate_csx!(shot, k, nu_ou, nu_eu, nu_ol, nu_el; tid)

    dR0x = evaluate_inbounds(shot.R0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dZ0x = evaluate_inbounds(shot.Z0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dϵx = evaluate_inbounds(shot.ϵfe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dκx = evaluate_inbounds(shot.κfe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dc0x = evaluate_inbounds(shot.c0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    evaluate_dcsx!(shot, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el; tid)

    Jf(θ) = f(θ) * MillerExtendedHarmonic.Jacobian(θ, R0x, ϵx, κx, c0x, shot._cx[tid], shot._sx[tid], dR0x, dZ0x, dϵx, dκx, dc0x, shot._dcx[tid], shot._dsx[tid])
    return twopi * trapa(Jf) / Vprime
end

function fsa_invR2(shot, ρ; tid = Threads.threadid())
    k, nu_ou, nu_eu, nu_ol, nu_el = compute_bases(shot.ρ, ρ)
    R0x = evaluate_inbounds(shot.R0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ϵx = evaluate_inbounds(shot.ϵfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    c0x = evaluate_inbounds(shot.c0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    evaluate_csx!(shot, k, nu_ou, nu_eu, nu_ol, nu_el; tid)
    ax = R0x * ϵx

    f(θ) = MillerExtendedHarmonic.R_MXH(θ, R0x, c0x, shot._cx[tid], shot._sx[tid], ax) ^ -2

    return FSA(f, shot, ρ)
end

function fsa_invR(shot, ρ; tid = Threads.threadid())
    k, nu_ou, nu_eu, nu_ol, nu_el = compute_bases(shot.ρ, ρ)
    R0x = evaluate_inbounds(shot.R0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ϵx = evaluate_inbounds(shot.ϵfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    c0x = evaluate_inbounds(shot.c0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    evaluate_csx!(shot, k, nu_ou, nu_eu, nu_ol, nu_el; tid)
    ax = R0x * ϵx

    f(θ) = MillerExtendedHarmonic.R_MXH(θ, R0x, c0x, shot._cx[tid], shot._sx[tid], ax) ^ -1

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

function Ip(shot; ε = 1e-6)
    coeffs = Vector{typeof(shot.ρ[1])}(undef, 2*length(shot.ρ))
    for (i, x) in enumerate(shot.ρ)
        # BCL 4/26/23: I'd like to use ForwardDiff here, but the use of intermediate arrays
        #              in evaluate_csx!() prevents that
        xp = x==shot.ρ[end] ? 1.0 : x + ε * (shot.ρ[i+1] - shot.ρ[i])
        xm = x==shot.ρ[1]   ? 0.0 : x - ε * (shot.ρ[i] - shot.ρ[i-1])
        coeffs[2i-1] = (Vprime(shot, xp) - Vprime(shot, xm)) / (xp - xm)
        coeffs[2i] = Vprime(shot, x)
    end
    Vp = FE_rep(shot.ρ, coeffs)
    if shot.Jt_R !== nothing
        f1(x) = Vp(x) * shot.Jt_R(x)
        return quadgk(f1, 0.0, 1.0)[1] / twopi
    elseif shot.Jt !== nothing
        invR = FE_fsa(shot, fsa_invR)
        f2(x) = Vp(x) * shot.Jt(x) * invR(x)
        return quadgk(f2, 0.0, 1.0)[1] / twopi
    else
        invR2 = FE_fsa(shot, fsa_invR2)
        if shot.dP_dψ !== nothing
            f3(x) = - Vp(x) * (shot.dP_dψ(x) + invR2(x) * shot.F_dF_dψ(x) / μ₀)
            return quadgk(f3, 0.0, 1.0)[1]
        else
            Pp(x) = D(shot.P, x) / dψ_dρ(shot, x)
            f4(x) = - Vp(x) * (Pp(x) + invR2(x) * shot.F_dF_dψ(x) / μ₀)
            return quadgk(f4, 0.0, 1.0)[1]
        end
    end
    return 0.0
end

function Ip_ffp(shot; ε = 1e-6)
    (shot.F_dF_dψ === nothing) && return 0.0

    coeffs = Vector{typeof(shot.ρ[1])}(undef, 2*length(shot.ρ))
    for (i, x) in enumerate(shot.ρ)
        # BCL 4/26/23: I'd like to use ForwardDiff here, but the use of intermediate arrays
        #              in evaluate_csx!() prevents that
        xp = x==shot.ρ[end] ? 1.0 : x + ε * (shot.ρ[i+1] - shot.ρ[i])
        xm = x==shot.ρ[1]   ? 0.0 : x - ε * (shot.ρ[i] - shot.ρ[i-1])
        coeffs[2i-1] = (Vprime(shot, xp) - Vprime(shot, xm)) / (xp - xm)
        coeffs[2i] = Vprime(shot, x)
    end
    Vp = FE_rep(shot.ρ, coeffs)

    invR2 = FE_fsa(shot, fsa_invR2)
    f(x) = - Vp(x) * invR2(x) * shot.F_dF_dψ(x) / μ₀
    return quadgk(f, 0.0, 1.0)[1]
end
