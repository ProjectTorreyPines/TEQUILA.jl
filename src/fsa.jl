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

function fsa_trapa(J::F1, f::F2; min_level=3, max_level=20, tol::Real=eps(typeof(1.0))) where {F1, F2}
    j = J(0.0)
    int_Jf = twopi * j * f(0.0)
    int_J  = twopi * j
    for l in 1:max_level
        dx = twopi / 2^l
        X = range(0, twopi, 2^l + 1)[2:2:end]
        int2_Jf, int2_J = dx .* sum(x -> Jf_J(x, J, f), X)
        int_Jf = 0.5 * int_Jf + int2_Jf
        int_J  = 0.5 * int_J  + int2_J
        if l >= min_level
            abs(int_Jf <= tol) && break
            (abs(int2_Jf / int_Jf - 0.5) <= tol) && (abs(int2_J / int_J - 0.5) <= tol) && break
        end
    end

    return int_Jf / int_J
end

function Vprime(shot::F1, ρ::Real; tid = Threads.threadid()) where {F1<:Shot}
    k, nu_ou, nu_eu, nu_ol, nu_el, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el = compute_both_bases(shot.ρ, ρ)
    R0x = evaluate_inbounds(shot.R0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ϵx = evaluate_inbounds(shot.ϵfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    κx = evaluate_inbounds(shot.κfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    c0x = evaluate_inbounds(shot.c0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    cx, sx = evaluate_csx!(shot, k, nu_ou, nu_eu, nu_ol, nu_el; tid)

    dR0x = evaluate_inbounds(shot.R0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dZ0x = evaluate_inbounds(shot.Z0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dϵx = evaluate_inbounds(shot.ϵfe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dκx = evaluate_inbounds(shot.κfe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dc0x = evaluate_inbounds(shot.c0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dcx, dsx = evaluate_dcsx!(shot, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el; tid)

    J = θ -> MillerExtendedHarmonic.Jacobian(θ, R0x, ϵx, κx, c0x, cx, sx, dR0x, dZ0x, dϵx, dκx, dc0x, dcx, dsx)
    return twopi * trapa(J)
end

function FSA(f::F1, shot::F2, ρ::Real; tid = Threads.threadid()) where {F1, F2<:Shot}

    ρ == 0.0 && return f(0.0)

    k, nu_ou, nu_eu, nu_ol, nu_el, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el = compute_both_bases(shot.ρ, ρ)
    R0x = evaluate_inbounds(shot.R0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ϵx = evaluate_inbounds(shot.ϵfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    κx = evaluate_inbounds(shot.κfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    c0x = evaluate_inbounds(shot.c0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    cx, sx = evaluate_csx!(shot, k, nu_ou, nu_eu, nu_ol, nu_el; tid)

    dR0x = evaluate_inbounds(shot.R0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dZ0x = evaluate_inbounds(shot.Z0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dϵx = evaluate_inbounds(shot.ϵfe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dκx = evaluate_inbounds(shot.κfe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dc0x = evaluate_inbounds(shot.c0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dcx, dsx = evaluate_dcsx!(shot, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el; tid)

    J  = θ -> MillerExtendedHarmonic.Jacobian(θ, R0x, ϵx, κx, c0x, cx, sx, dR0x, dZ0x, dϵx, dκx, dc0x, dcx, dsx)
    return fsa_trapa(J, f)
end

function FSA(f::F1, shot::F2, ρ::Real, Vprime::F3; tid = Threads.threadid()) where {F1, F2<:Shot, F3<:FE_rep}

    ρ == 0.0 && return f(0.0)

    k, nu_ou, nu_eu, nu_ol, nu_el, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el = compute_both_bases(shot.ρ, ρ)
    R0x = evaluate_inbounds(shot.R0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ϵx = evaluate_inbounds(shot.ϵfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    κx = evaluate_inbounds(shot.κfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    c0x = evaluate_inbounds(shot.c0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    cx, sx = evaluate_csx!(shot, k, nu_ou, nu_eu, nu_ol, nu_el; tid)

    dR0x = evaluate_inbounds(shot.R0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dZ0x = evaluate_inbounds(shot.Z0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dϵx = evaluate_inbounds(shot.ϵfe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dκx = evaluate_inbounds(shot.κfe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dc0x = evaluate_inbounds(shot.c0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dcx, dsx = evaluate_dcsx!(shot, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el; tid)

    Vp = evaluate_inbounds(Vprime, k, nu_ou, nu_eu, nu_ol, nu_el)
    Jf = θ -> f(θ) * MillerExtendedHarmonic.Jacobian(θ, R0x, ϵx, κx, c0x, cx, sx, dR0x, dZ0x, dϵx, dκx, dc0x, dcx, dsx)

    return twopi * trapa(Jf) / Vp
end

function FSA(f::F1, shot::F2, ρ::Real, Vprime::Real; tid = Threads.threadid()) where {F1, F2<:Shot}

    ρ == 0.0 && return f(0.0)

    k, nu_ou, nu_eu, nu_ol, nu_el, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el = compute_both_bases(shot.ρ, ρ)
    R0x = evaluate_inbounds(shot.R0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ϵx = evaluate_inbounds(shot.ϵfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    κx = evaluate_inbounds(shot.κfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    c0x = evaluate_inbounds(shot.c0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    cx, sx = evaluate_csx!(shot, k, nu_ou, nu_eu, nu_ol, nu_el; tid)

    dR0x = evaluate_inbounds(shot.R0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dZ0x = evaluate_inbounds(shot.Z0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dϵx = evaluate_inbounds(shot.ϵfe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dκx = evaluate_inbounds(shot.κfe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dc0x = evaluate_inbounds(shot.c0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dcx, dsx = evaluate_dcsx!(shot, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el; tid)

    Jf = θ -> f(θ) * MillerExtendedHarmonic.Jacobian(θ, R0x, ϵx, κx, c0x, cx, sx, dR0x, dZ0x, dϵx, dκx, dc0x, dcx, dsx)
    return twopi * trapa(Jf) / Vprime
end

function fsa_invR2(shot::F1, ρ; tid = Threads.threadid()) where {F1<:Shot}
    k, nu_ou, nu_eu, nu_ol, nu_el = compute_bases(shot.ρ, ρ)
    R0x = evaluate_inbounds(shot.R0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ϵx = evaluate_inbounds(shot.ϵfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    c0x = evaluate_inbounds(shot.c0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    cx, sx = evaluate_csx!(shot, k, nu_ou, nu_eu, nu_ol, nu_el; tid)
    ax = R0x * ϵx

    f = θ -> MillerExtendedHarmonic.R_MXH(θ, R0x, c0x, cx, sx, ax) ^ -2

    return FSA(f, shot, ρ)
end

function fsa_invR(shot::F1, ρ; tid = Threads.threadid()) where {F1<:Shot}
    k, nu_ou, nu_eu, nu_ol, nu_el = compute_bases(shot.ρ, ρ)
    R0x = evaluate_inbounds(shot.R0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ϵx = evaluate_inbounds(shot.ϵfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    c0x = evaluate_inbounds(shot.c0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    cx, sx = evaluate_csx!(shot, k, nu_ou, nu_eu, nu_ol, nu_el; tid)
    ax = R0x * ϵx

    f = θ -> MillerExtendedHarmonic.R_MXH(θ, R0x, c0x, cx, sx, ax) ^ -1

    return FSA(f, shot, ρ)
end

function FiniteElementHermite.FE_rep(shot::F1, f::F2, coeffs = Vector{typeof(shot.ρ[1])}(undef, 2*length(shot.ρ)); ε = 1e-6) where {F1<:Shot, F2}
    for (i, x) in enumerate(shot.ρ)
        # BCL 4/26/23: I'd like to use ForwardDiff here, but the use of intermediate arrays
        #              in evaluate_csx!() prevents that
        # BCL 6/2/23: Can use ForwardDiff now but it gives slightly different results,
        #             maybe due to issues at boundary? Don't know what is best but
        #             going with ForwardDiff
        g = x ->  f(shot, x)
        if x == 0.0
            coeffs[2i-1] = ForwardDiff.derivative(g, 1e-12)
        else
            coeffs[2i-1] = ForwardDiff.derivative(g, x)
        end
        coeffs[2i] = g(x)
    end
    return FE_rep(shot.ρ, coeffs)
end

function FE_coeffs!(Y::FE_rep, shot::F1, f::F2; ε::Real = 1e-6, derivative::Symbol=:auto) where {F1<:Shot, F2}
    @assert derivative in (:auto, :finite)
    for (i, x) in enumerate(Y.x)
        g = x ->  f(shot, x)
        if derivative === :auto
            if x == 0.0
                Y.coeffs[2i-1] = ForwardDiff.derivative(g, 1e-12)
            else
                Y.coeffs[2i-1] = ForwardDiff.derivative(g, x)
            end
        else
            xp = x==Y.x[end] ? x : x + ε * (Y.x[i+1] - Y.x[i])
            xm = x==Y.x[1]   ? x : x - ε * (Y.x[i] - Y.x[i-1])
            Y.coeffs[2i-1] = (f(shot, xp) - f(shot, xm)) / (xp - xm)
        end

        Y.coeffs[2i] = g(x)
    end
    return Y
end

function _int_J_invR2(shot::F1, ρ::Real; tid = Threads.threadid()) where {F1<:Shot}
    k, nu_ou, nu_eu, nu_ol, nu_el, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el = compute_both_bases(shot.ρ, ρ)
    R0x = evaluate_inbounds(shot.R0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ϵx = evaluate_inbounds(shot.ϵfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    κx = evaluate_inbounds(shot.κfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    c0x = evaluate_inbounds(shot.c0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    cx, sx = evaluate_csx!(shot, k, nu_ou, nu_eu, nu_ol, nu_el; tid)
    ax = R0x * ϵx

    dR0x = evaluate_inbounds(shot.R0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dZ0x = evaluate_inbounds(shot.Z0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dϵx = evaluate_inbounds(shot.ϵfe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dκx = evaluate_inbounds(shot.κfe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dc0x = evaluate_inbounds(shot.c0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dcx, dsx = evaluate_dcsx!(shot, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el; tid)

    J_invR2  = θ -> (MillerExtendedHarmonic.Jacobian(θ, R0x, ϵx, κx, c0x, cx, sx, dR0x, dZ0x, dϵx, dκx, dc0x, dcx, dsx) *
                     MillerExtendedHarmonic.R_MXH(θ, R0x, c0x, cx, sx, ax) ^ -2)

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

function toroidal_flux!(Φ::Vector{<:Real}, shot::F1, ρs::AbstractVector{<:Real}) where {F1<:Shot}
    @assert length(Φ) === length(ρs)
    invR = FE_rep(shot, fsa_invR)
    invR2 = FE_rep(shot, fsa_invR2)
    Vp = FE_rep(shot, Vprime)
    f = x -> Fpol(shot, x; invR, invR2) * Vp(x) * invR2(x)
    for k in eachindex(ρs)[2:end]
        Φ[k] = Φ[k-1] + quadgk(f, ρs[k-1], ρs[k])[1] / twopi
    end
    return Φ
end

function rho_tor_norm(shot::F1, ρ::Real) where{F1<:Shot}
    Φ = toroidal_flux(ρ, shot.F, shot.Vp, shot.invR2)
    Φ0 = toroidal_flux(1.0, shot.F, shot.Vp, shot.invR2)
    return sqrt(abs(Φ / Φ0)) # abs prevents -0.0
end

function rho_tor_norm(shot::F1) where{F1<:Shot}
    ρtor = zero(shot.ρ)
    return rho_tor_norm!(ρtor, shot)
end

function rho_tor_norm!(ρtor::Vector{<:Real}, shot::F1) where{F1<:Shot}
    @assert length(ρtor) === length(shot.ρ)
    toroidal_flux!(ρtor, shot, shot.ρ)
    @. ρtor = sqrt(ρtor / ρtor[end])
    ρtor[1] = 0.0
    ρtor[end] = 1.0
    return ρtor
end

function set_FSAs!(shot)
    FE_coeffs!(shot.Vp, shot, Vprime; derivative=:auto)
    FE_coeffs!(shot.invR, shot, fsa_invR; derivative=:auto)
    FE_coeffs!(shot.invR2, shot, fsa_invR2; derivative=:auto)
    FE_coeffs!(shot.F, shot, (shot, x) -> Fpol(shot, x; shot.invR, shot.invR2); derivative=:finite)
    FE_coeffs!(shot.ρtor, shot, rho_tor_norm; derivative=:finite)
    return shot
end

function Ψ(shot)
    @views Psi = FE_rep(shot.ρ, shot.C[:,1])
    return Psi
end

function Ip(shot::F1; ε::Real = 1e-6) where {F1<:Shot}
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
        f3 = x ->  - Vp(x) * (Pp(x) + invR2(x) * shot.F_dF_dψ(x) / μ₀)
        return quadgk(f3, 0.0, 1.0)[1]
    end
    return 0.0
end

function Ip_ffp(shot::F1; ε::Real = 1e-6) where {F1<:Shot}
    (shot.F_dF_dψ === nothing) && return 0.0

    Vp = FE_rep(shot, Vprime)
    invR2 = FE_rep(shot, fsa_invR2)
    f = x -> - Vp(x) * invR2(x) * shot.F_dF_dψ(x) / μ₀
    return quadgk(f, 0.0, 1.0)[1]
end



function flux_surfaces(shot::Shot)

    # fw = IMAS.first_wall(IMAS.top_dd(eqt).wall)
    # eqt2d = findfirst(:rectangular, eqt.profiles_2d)
    # r, z, PSI_interpolant = ψ_interpolant(eqt2d)
    # PSI = eqt2d.psi

    psi_sign = sign(eqt.profiles_1d.psi[end] - eqt.profiles_1d.psi[1])
    R0 = shot.surfaces[1, end]
    B0 = shot.Fbnd / R0

    # # ensure certain global quantities are consistent with 1d profiles by making them expressions
    # for field in (:psi_boundary, :psi_axis, :q_95, :q_axis, :q_min)
    #     empty!(eqt.global_quantities, field)
    # end

    # accurately find magnetic axis and lcfs and scale psi accordingly
    RA, ZA = shot.surfaces[1, 1], shot.surfaces[2, 1]
    #eqt.global_quantities.magnetic_axis.r, eqt.global_quantities.magnetic_axis.z = RA, ZA
    psi_axis = ψ₀(shot)
    psi_boundary = 0.0
    psi = shot.C[2:2:end,1]
    #eqt.profiles_1d.psi = (eqt.profiles_1d.psi .- eqt.profiles_1d.psi[1]) ./ (eqt.profiles_1d.psi[end] - eqt.profiles_1d.psi[1]) .* (psi_boundary - psi_axis) .+ psi_axis

    b_field_average = zero(psi)
    b_field_max = zero(psi)
    b_field_min = zero(psi)
    elongation = zero(psi)
    triangularity_lower = zero(psi)
    triangularity_upper = zero(psi)
    squareness_lower_inner = zero(psi)
    squareness_lower_outer = zero(psi)
    squareness_upper_inner = zero(psi)
    squareness_upper_outer = zero(psi)
    r_inboard = zero(psi)
    r_outboard = zero(psi)
    q = zero(psi)
    surface = zero(psi)
    dvolume_dpsi = zero(psi)
    j_tor = zero(psi)
    gm1 = zero(psi)
    gm2 = zero(psi)
    gm4 = zero(psi)
    gm5 = zero(psi)
    gm8 = zero(psi)
    gm9 = zero(psi)
    gm10 = zero(psi)
    fsa_bp = zero(psi)
    trapped_fraction = zero(psi)

    # PR = Vector{T}[]
    # PZ = Vector{T}[]
    # LL = Vector{T}[]
    # FLUXEXPANSION = Vector{T}[]
    # INT_FLUXEXPANSION_DL = zeros(T, length(eqt.profiles_1d.psi))
    # BPL = zeros(T, length(eqt.profiles_1d.psi))

    for k in length(psi):-1:1
        psi_level0 = psi[k]

        if k == 1 # on axis flux surface is a synthetic one
            # elongation[1] = elongation[2] - (elongation[3] - elongation[2])
            # triangularity_upper[1] = 0.0
            # triangularity_lower[1] = 0.0

            # a = (eqt.profiles_1d.r_outboard[2] - eqt.profiles_1d.r_inboard[2]) / 100.0
            # b = eqt.profiles_1d.elongation[1] * a

            # t = range(0, 2π, 17)
            # pr = cos.(t) .* a .+ eqt.global_quantities.magnetic_axis.r
            # pz = sin.(t) .* b .+ eqt.global_quantities.magnetic_axis.z

            # # Extrema on array indices
            # (imaxr, iminr, imaxz, iminz, r_at_max_z, max_z, r_at_min_z, min_z, z_at_max_r, max_r, z_at_min_r, min_r) = fluxsurface_extrema(pr, pz)

        else  # other flux surfaces

            # Extrema on array indices
            (imaxr, iminr, imaxz, iminz, r_at_max_z, max_z, r_at_min_z, min_z, z_at_max_r, max_r, z_at_min_r, min_r) = fluxsurface_extrema(pr, pz)

            # accurate geometric quantities by finding geometric extrema as optimization problem

            # plasma boundary information
            if k == length(eqt.profiles_1d.psi)
                eqt.boundary.outline.r = pr
                eqt.boundary.outline.z = pz
            end
        end

        @views R0, Z0, ϵ, κ, c0, c, s = unflatten_view(shot.surfaces[:, k])
        eqt.profiles_1d.r_outboard[k] = R0 * (1.0 + ϵ)
        eqt.profiles_1d.r_inboard[k]  = R0 * (1.0 - ϵ)

        # miller geometric coefficients
        _, _, _, δu, δl, ζou, ζol, ζil, ζiu = miller_R_a_κ_δ_ζ(pr, pz, r_at_max_z, max_z, r_at_min_z, min_z, z_at_max_r, max_r, z_at_min_r, min_r)
        eqt.profiles_1d.elongation[k] = κ
        eqt.profiles_1d.triangularity_upper[k] = δu
        eqt.profiles_1d.triangularity_lower[k] = δl
        eqt.profiles_1d.squareness_lower_outer[k] = ζol
        eqt.profiles_1d.squareness_upper_outer[k] = ζou
        eqt.profiles_1d.squareness_lower_inner[k] = ζil
        eqt.profiles_1d.squareness_upper_inner[k] = ζiu

        # poloidal magnetic field (with sign)
        Br, Bz = Br_Bz(PSI_interpolant, pr, pz)
        Bp2 = Br .^ 2.0 .+ Bz .^ 2.0
        Bp_abs = sqrt.(Bp2)
        Bp = (
            Bp_abs .*
            sign.((pz .- eqt.global_quantities.magnetic_axis.z) .* Br .- (pr .- eqt.global_quantities.magnetic_axis.r) .* Bz)
        )

        # flux expansion
        dl = vcat(0.0, sqrt.(diff(pr) .^ 2 + diff(pz) .^ 2))
        ll = cumsum(dl)
        fluxexpansion = 1.0 ./ Bp_abs
        int_fluxexpansion_dl = trapz(ll, fluxexpansion)
        Bpl = trapz(ll, Bp)

        # save flux surface coordinates for later use
        pushfirst!(PR, pr)
        pushfirst!(PZ, pz)
        pushfirst!(LL, ll)
        pushfirst!(FLUXEXPANSION, fluxexpansion)
        INT_FLUXEXPANSION_DL[k] = int_fluxexpansion_dl
        BPL[k] = Bpl

        # trapped fraction
        Bt = eqt.profiles_1d.f[k] ./ pr
        Btot = sqrt.(Bp2 .+ Bt .^ 2)
        Bmin = minimum(Btot)
        Bmax = maximum(Btot)
        Bratio = Btot ./ Bmax
        avg_Btot = flxAvg(Btot, ll, fluxexpansion, int_fluxexpansion_dl)
        avg_Btot2 = flxAvg(Btot .^ 2, ll, fluxexpansion, int_fluxexpansion_dl)
        hf = flxAvg((1.0 .- sqrt.(1.0 .- Bratio) .* (1.0 .+ Bratio ./ 2.0)) ./ Bratio .^ 2, ll, fluxexpansion, int_fluxexpansion_dl)
        h = avg_Btot / Bmax
        h2 = avg_Btot2 / Bmax^2
        ftu = 1.0 - h2 / (h^2) * (1.0 - sqrt(1.0 - h) * (1.0 + 0.5 * h))
        ftl = 1.0 - h2 * hf
        eqt.profiles_1d.trapped_fraction[k] = 0.75 * ftu + 0.25 * ftl

        # Bavg
        eqt.profiles_1d.b_field_average[k] = avg_Btot

        # Bmax
        eqt.profiles_1d.b_field_max[k] = Bmax

        # Bmin
        eqt.profiles_1d.b_field_min[k] = Bmin

        # gm1 = <1/R^2>
        eqt.profiles_1d.gm1[k] = flxAvg(1.0 ./ pr .^ 2, ll, fluxexpansion, int_fluxexpansion_dl)

        # gm4 = <1/B^2>
        eqt.profiles_1d.gm4[k] = flxAvg(1.0 ./ Btot .^ 2, ll, fluxexpansion, int_fluxexpansion_dl)

        # gm5 = <B^2>
        eqt.profiles_1d.gm5[k] = avg_Btot2

        # gm8 = <R>
        eqt.profiles_1d.gm8[k] = flxAvg(pr, ll, fluxexpansion, int_fluxexpansion_dl)

        # gm9 = <1/R>
        eqt.profiles_1d.gm9[k] = flxAvg(1.0 ./ pr, ll, fluxexpansion, int_fluxexpansion_dl)

        # gm10 = <R^2>
        eqt.profiles_1d.gm10[k] = flxAvg(pr .^ 2, ll, fluxexpansion, int_fluxexpansion_dl)

        # fsa_bp = <Bp>
        eqt.profiles_1d.fsa_bp[k] = flxAvg(Bp, ll, fluxexpansion, int_fluxexpansion_dl)

        # j_tor = <j_tor/R> / <1/R> [A/m²]
        eqt.profiles_1d.j_tor[k] =
            (
                -(eqt.profiles_1d.dpressure_dpsi[k] + eqt.profiles_1d.f_df_dpsi[k] * eqt.profiles_1d.gm1[k] / constants.μ_0) *
                (2π)
            ) / eqt.profiles_1d.gm9[k]

        # dvolume_dpsi
        eqt.profiles_1d.dvolume_dpsi[k] = sign(eqt.profiles_1d.fsa_bp[k]) * int_fluxexpansion_dl

        # surface area
        eqt.profiles_1d.surface[k] = 2π * sum(pr .* dl)

        # q
        eqt.profiles_1d.q[k] = eqt.profiles_1d.dvolume_dpsi[k] .* eqt.profiles_1d.f[k] .* eqt.profiles_1d.gm1[k] ./ (2π)

        # quantities calculated on the last closed flux surface
        if k == length(eqt.profiles_1d.psi)
            # perimeter
            eqt.global_quantities.length_pol = ll[end]
        end
    end

    # area
    eqt.profiles_1d.area = cumtrapz(eqt.profiles_1d.psi, eqt.profiles_1d.dvolume_dpsi .* eqt.profiles_1d.gm9) ./ 2π

    # volume
    eqt.profiles_1d.volume = cumtrapz(eqt.profiles_1d.psi, eqt.profiles_1d.dvolume_dpsi)

    # phi
    eqt.profiles_1d.phi = cumtrapz(eqt.profiles_1d.volume, eqt.profiles_1d.f .* eqt.profiles_1d.gm1) / (2π)

    # rho_tor_norm
    rho = sqrt.(abs.(eqt.profiles_1d.phi ./ (π * B0)))
    rho_meters = rho[end]
    eqt.profiles_1d.rho_tor = rho
    eqt.profiles_1d.rho_tor_norm = rho ./ rho_meters

    # phi 2D
    eqt2d.phi = interp1d(eqt.profiles_1d.psi * psi_sign, eqt.profiles_1d.phi, :cubic).(eqt2d.psi * psi_sign)

    # rho 2D in meters
    RHO = sqrt.(abs.(eqt2d.phi ./ (π * B0)))

    # gm2: <∇ρ²/R²>
    if false
        RHO_interpolant = Interpolations.cubic_spline_interpolation((r, z), RHO)
        for k in 1:length(eqt.profiles_1d.psi)
            tmp = [Interpolations.gradient(RHO_interpolant, PR[k][j], PZ[k][j]) for j in 1:length(PR[k])]
            dPHI2 = [j[1] .^ 2.0 .+ j[2] .^ 2.0 for j in tmp]
            eqt.profiles_1d.gm2[k] = flxAvg(dPHI2 ./ PR[k] .^ 2.0, LL[k], FLUXEXPANSION[k], INT_FLUXEXPANSION_DL[k])
        end
    else
        dRHOdR, dRHOdZ = gradient(collect(r), collect(z), RHO)
        dPHI2_interpolant = Interpolations.cubic_spline_interpolation((r, z), dRHOdR .^ 2.0 .+ dRHOdZ .^ 2.0)
        for k in 1:length(eqt.profiles_1d.psi)
            dPHI2 = dPHI2_interpolant.(PR[k], PZ[k])
            eqt.profiles_1d.gm2[k] = flxAvg(dPHI2 ./ PR[k] .^ 2.0, LL[k], FLUXEXPANSION[k], INT_FLUXEXPANSION_DL[k])
        end
    end
    eqt.profiles_1d.gm2[1] = interp1d(eqt.profiles_1d.psi[2:end] * psi_sign, eqt.profiles_1d.gm2[2:end], :cubic).(eqt.profiles_1d.psi[1] * psi_sign)

    # ip
    eqt.global_quantities.ip = trapz(eqt.profiles_1d.area, eqt.profiles_1d.j_tor)
    # eqt.global_quantities.ip = trapz(eqt.profiles_1d.volume, eqt.profiles_1d.j_tor.*eqt.profiles_1d.gm9) / (2π) # equivalent

    # Geometric major and minor radii
    Rgeo = (eqt.profiles_1d.r_outboard[end] + eqt.profiles_1d.r_inboard[end]) / 2.0
    a = (eqt.profiles_1d.r_outboard[end] - eqt.profiles_1d.r_inboard[end]) / 2.0

    # vacuum magnetic field at the geometric center
    Btvac = B0 * R0 / Rgeo

    # average poloidal magnetic field
    Bpave = eqt.global_quantities.ip * constants.μ_0 / eqt.global_quantities.length_pol

    # li
    Bp2v = trapz(eqt.profiles_1d.psi, BPL)
    eqt.global_quantities.li_3 = 2.0 * Bp2v / Rgeo / (eqt.global_quantities.ip * constants.μ_0)^2

    # beta_tor
    avg_press = volume_integrate(eqt, eqt.profiles_1d.pressure) / eqt.profiles_1d.volume[end]
    eqt.global_quantities.beta_tor = abs(avg_press / (Btvac^2 / 2.0 / constants.μ_0))

    # beta_pol
    eqt.global_quantities.beta_pol = abs(avg_press / (Bpave^2 / 2.0 / constants.μ_0))

    # beta_normal
    ip = eqt.global_quantities.ip / 1e6
    eqt.global_quantities.beta_normal = eqt.global_quantities.beta_tor / abs(ip / a / Btvac) * 100

    # find quantities on separatrix
    find_x_point!(eqt)
    find_strike_points!(eqt)

    # secondary separatrix
    if length(eqt.boundary.x_point) > 1
        psi2nd = find_psi_2nd_separatrix(eqt)
        tmp, _ = flux_surface(r, z, PSI, RA, ZA, fw.r, fw.z, psi2nd, :encircling)
        if !isempty(tmp)
            (pr2nd, pz2nd) = tmp[1]
            eqt.boundary_secondary_separatrix.outline.r = pr2nd
            eqt.boundary_secondary_separatrix.outline.z = pz2nd
        end
    end

    return eqt
end