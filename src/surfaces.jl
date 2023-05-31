function concentric_surface(x::Real, boundary::MXH; Raxis::Real = boundary.R0, Zaxis::Real = boundary.Z0)
    surface = deepcopy(boundary)
    return scale_surface!(surface, x; Raxis, Zaxis)
end

function concentric_surface!(surface::MXH, x::Real, boundary::MXH; Raxis::Real = boundary.R0, Zaxis::Real = boundary.Z0)
    copy_MXH!(surface, boundary)
    return scale_surface!(surface, x; Raxis, Zaxis)
end

function scale_surface!(surface::MXH, x::Real; Raxis::Real = surface.R0, Zaxis::Real = surface.Z0)
    # these go to zero as you go to axis
    a = surface.R0 * surface.ϵ * x
    surface.R0 = x * surface.R0 + (1.0 - x) * Raxis
    surface.Z0 = x * surface.Z0 + (1.0 - x) * Zaxis
    surface.ϵ  = a / surface.R0
    if x < 1.0
        surface.c .*= x
        surface.s .*= x
    end
    return surface
end

function surfaces_FE(shot::Shot)
    surfaces_FE(shot.ρ, shot.surfaces)
end

function surfaces_FE(ρ:: AbstractVector{<:Real}, surfaces:: AbstractVector{<:MXH} )

    N = length(ρ)
    M_mxh = length(surfaces[1].c)
    R0s = zeros(N)
    Z0s = zeros(N)
    ϵs  = zeros(N)
    κs  = zeros(N)
    c0s = zeros(N)
    cs  = zeros(N, M_mxh)
    ss  = zeros(N, M_mxh)

    for (i, surface) in enumerate(surfaces)
        R0s[i] = surface.R0
        Z0s[i] = surface.Z0
        ϵs[i] = surface.ϵ
        κs[i] = surface.κ
        c0s[i] = surface.c0
        cs[i,:] = surface.c
        ss[i,:] = surface.s
    end

    rtype = typeof(ρ[1])

    R0fe = FE(ρ, rtype.(R0s))
    Z0fe = FE(ρ, rtype.(Z0s))
    ϵfe = FE(ρ, rtype.(ϵs))
    κfe = FE(ρ, rtype.(κs))
    c0fe = FE(ρ, rtype.(c0s))

    cfe = Vector{typeof(R0fe)}(undef, M_mxh)
    sfe = Vector{typeof(R0fe)}(undef, M_mxh)
    for m in 1:M_mxh
        @views cfe[m] = FE(ρ, rtype.(cs[:, m]))
        @views sfe[m] = FE(ρ, rtype.(ss[:, m]))
    end

    return R0fe, Z0fe, ϵfe, κfe, c0fe, cfe, sfe
end

function surfaces_FE(ρ:: AbstractVector{<:Real}, surfaces:: AbstractMatrix{<:Real} )

    rtype = typeof(ρ[1])

    @views R0fe = FE(ρ, rtype.(surfaces[1, :]))
    @views Z0fe = FE(ρ, rtype.(surfaces[2, :]))
    @views ϵfe  = FE(ρ, rtype.(surfaces[3, :]))
    @views κfe  = FE(ρ, rtype.(surfaces[4, :]))
    @views c0fe = FE(ρ, rtype.(surfaces[5, :]))

    M = (size(surfaces,1) - 5) ÷ 2

    cfe = Vector{typeof(R0fe)}(undef, M)
    sfe = Vector{typeof(R0fe)}(undef, M)
    for m in 1:M
        @views cfe[m] = FE(ρ, rtype.(surfaces[5 + m, :]))
        @views sfe[m] = FE(ρ, rtype.(surfaces[5 + m + M, :]))
    end

    return R0fe, Z0fe, ϵfe, κfe, c0fe, cfe, sfe
end

function Δ(shot, ρ, R, Z; return_θ = false)
    k, nu_ou, nu_eu, nu_ol, nu_el = TEQUILA.compute_bases(shot.ρ, ρ)
    R0x = TEQUILA.evaluate_inbounds(shot.R0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    Z0x = TEQUILA.evaluate_inbounds(shot.Z0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ϵx = TEQUILA.evaluate_inbounds(shot.ϵfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    κx = TEQUILA.evaluate_inbounds(shot.κfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    c0x = TEQUILA.evaluate_inbounds(shot.c0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    TEQUILA.evaluate_csx!(shot, k, nu_ou, nu_eu, nu_ol, nu_el)

    ax = R0x * ϵx
    bx = κx * ax

    if bx == 0.0
        θ = 0.0
    else
        aa = min(1.0, max(-1.0, (Z0x - Z) / bx))
        θ = asin(aa)
    end
    signθ = (θ < 0.0) ? -1.0 : 1.0
    minmax = (signθ > 0.0) ? :min : :max
    R_at_Zext = MillerExtendedHarmonic.R_at_Zext(minmax, R0x, c0x, shot._cx, shot._sx, ax)

    (R < R_at_Zext) && (θ = signθ * π - θ)

    return_θ && return θ

    Rx = MillerExtendedHarmonic.R_MXH(θ, R0x, c0x, shot._cx, shot._sx, ax)
    Zx = MillerExtendedHarmonic.Z_MXH(θ, Z0x, κx, ax)
    Δ = R < R_at_Zext ? R - Rx : Rx - R
    Δ -= abs(Z-Zx)
    return Δ
end

function ρθ_RZ(shot, R, Z)
    f(x) = Δ(shot, x, R, Z)
    ρ = f(1.0) < 0.0 ? 1.0 : Roots.find_zero(f, (0,1), Roots.A42())
    θ = (ρ == 0.0) ? 0.0 : Δ(shot, ρ, R, Z; return_θ = true)
    return ρ, θ
end

function fnl(u, p::Tuple{<:Real, <:Real, <:Shot})
    r, z = R_Z(p[3], u[1], u[2])
    return @SVector[r - p[1], z - p[2]]
end
function Jnl(u, p::Tuple{<:Real, <:Real, <:Shot})
    R_ρ, R_θ, Z_ρ, Z_θ = TEQUILA.JacMat(p[3], u[1], u[2])
    return @SMatrix[R_ρ R_θ; Z_ρ Z_θ]
end

function ρθ_RZ2(shot, R, Z)
    fJ = NonlinearFunction{false, NonlinearSolve.SciMLBase.FullSpecialize}(fnl; jac=Jnl)
    u0 = @SVector[0.5, 0.0]
    p = (R, Z, shot)
    probN = NonlinearProblem{false}(fJ, u0, p)
    S = NonlinearSolve.solve(probN, SimpleNewtonRaphson())
    ρ, θ = S[1], S[2]
    return ρ, θ
end 

##########################################################
# BCL 8/24/22: THESE SHOULD ALL MAKE USE OF compute_bases
##########################################################

function evaluate_csx!(shot::Shot, k::Integer, nu_ou, nu_eu, nu_ol, nu_el)
    for m in eachindex(shot.cfe)
        shot._cx[m] = evaluate_inbounds(shot.cfe[m], k, nu_ou, nu_eu, nu_ol, nu_el)
        shot._sx[m] = evaluate_inbounds(shot.sfe[m], k, nu_ou, nu_eu, nu_ol, nu_el)
    end
    return shot
end

function evaluate_dcsx!(shot::Shot, k::Integer, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    for m in eachindex(shot.cfe)
        shot._dcx[m] =  evaluate_inbounds(shot.cfe[m], k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
        shot._dsx[m] =  evaluate_inbounds(shot.sfe[m], k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    end
    return shot
end

function MillerExtendedHarmonic.MXH(shot::Shot, ρ::Real)
    k, nu_ou, nu_eu, nu_ol, nu_el = compute_bases(shot.ρ, ρ)
    R0x = evaluate_inbounds(shot.R0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    Z0x = evaluate_inbounds(shot.Z0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ϵx = evaluate_inbounds(shot.ϵfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    κx = evaluate_inbounds(shot.κfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    c0x = evaluate_inbounds(shot.c0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    evaluate_csx!(shot, k, nu_ou, nu_eu, nu_ol, nu_el)
    return MXH(R0x, Z0x, ϵx, κx, c0x, shot._cx, shot._sx)
end


function Tr(shot::Shot, ρ::Real, θ::Real)
    k, nu_ou, nu_eu, nu_ol, nu_el = compute_bases(shot.ρ, ρ)
    c0 = evaluate_inbounds(shot.c0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    evaluate_csx!(shot, k, nu_ou, nu_eu, nu_ol, nu_el)
    return MillerExtendedHarmonic.Tr(θ, c0, shot._cx, shot._sx)
end

function dTr_dρ(shot::Shot, ρ::Real, θ::Real)
    k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el = compute_D_bases(shot.ρ, ρ)
    dc0 = evaluate_inbounds(shot.c0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    evaluate_dcsx!(shot, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    return MillerExtendedHarmonic.dTr_dρ(θ, dc0, shot._dcx, shot._dsx) # ρ derivative just passes through
end

function dTr_dθ(shot::Shot, ρ::Real, θ::Real)
    k, nu_ou, nu_eu, nu_ol, nu_el = compute_bases(shot.ρ, ρ)
    evaluate_csx!(shot, k, nu_ou, nu_eu, nu_ol, nu_el)
    return MillerExtendedHarmonic.dTr_dθ(θ, shot._cx, shot._sx)
end

function R(shot::Shot, ρ::Real, θ::Real)
    k, nu_ou, nu_eu, nu_ol, nu_el = compute_bases(shot.ρ, ρ)
    R0x = evaluate_inbounds(shot.R0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ϵx = evaluate_inbounds(shot.ϵfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    c0x = evaluate_inbounds(shot.c0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    evaluate_csx!(shot, k, nu_ou, nu_eu, nu_ol, nu_el)
    ax = R0x * ϵx

    return MillerExtendedHarmonic.R_MXH(θ, R0x, c0x, shot._cx, shot._sx, ax)
end

function Z(shot::Shot, ρ::Real, θ::Real)
    k, nu_ou, nu_eu, nu_ol, nu_el = compute_bases(shot.ρ, ρ)
    R0x = evaluate_inbounds(shot.R0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    Z0x = evaluate_inbounds(shot.Z0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ϵx = evaluate_inbounds(shot.ϵfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    κx = evaluate_inbounds(shot.κfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ax = R0x * ϵx

    return MillerExtendedHarmonic.Z_MXH(θ, Z0x, κx, ax)
end

function R_Z(shot::Shot, ρ::Real, θ::Real)
    k, nu_ou, nu_eu, nu_ol, nu_el = compute_bases(shot.ρ, ρ)
    R0x = evaluate_inbounds(shot.R0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    Z0x = evaluate_inbounds(shot.Z0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ϵx = evaluate_inbounds(shot.ϵfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    κx = evaluate_inbounds(shot.κfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    c0x = evaluate_inbounds(shot.c0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    evaluate_csx!(shot, k, nu_ou, nu_eu, nu_ol, nu_el)
    ax = R0x * ϵx

    R = MillerExtendedHarmonic.R_MXH(θ, R0x, c0x, shot._cx, shot._sx, ax)
    Z = MillerExtendedHarmonic.Z_MXH(θ, Z0x, κx, ax)
    return R, Z
end

function R_Z(R0fe::FE_rep, Z0fe::FE_rep, ϵfe::FE_rep, κfe::FE_rep, c0fe::FE_rep,
    cfe::AbstractVector{<:FE_rep}, sfe::AbstractVector{<:FE_rep}, ρ::Real, θ::Real)

    k, nu_ou, nu_eu, nu_ol, nu_el = compute_bases(R0fe.x, ρ)
    R0x = evaluate_inbounds(R0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    Z0x = evaluate_inbounds(Z0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ϵx = evaluate_inbounds(ϵfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    κx = evaluate_inbounds(κfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    c0x = evaluate_inbounds(c0fe, k, nu_ou, nu_eu, nu_ol, nu_el)

    ax = R0x * ϵx

    csx = 0.0
    @inbounds for m in eachindex(cfe)
        smx = evaluate_inbounds(sfe[m], k, nu_ou, nu_eu, nu_ol, nu_el)
        cmx = evaluate_inbounds(cfe[m], k, nu_ou, nu_eu, nu_ol, nu_el)
        csx += dot((smx, cmx), sincos(m * θ))
    end

    θr = θ + c0x + csx
    R = R0x + ax * cos(θr)
    Z = MillerExtendedHarmonic.Z_MXH(θ, Z0x, κx, ax)
    return R, Z
end

function dR_dρ(shot::Shot, ρ::Real, θ::Real)
    k, nu_ou, nu_eu, nu_ol, nu_el, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el = compute_both_bases(shot.ρ, ρ)
    R0x = evaluate_inbounds(shot.R0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ϵx = evaluate_inbounds(shot.ϵfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    c0x = evaluate_inbounds(shot.c0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    evaluate_csx!(shot, k, nu_ou, nu_eu, nu_ol, nu_el)

    dR0x = evaluate_inbounds(shot.R0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dϵx = evaluate_inbounds(shot.ϵfe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dc0x = evaluate_inbounds(shot.c0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    evaluate_dcsx!(shot, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)

    return MillerExtendedHarmonic.dR_dρ(θ, R0x, ϵx, c0x, shot._cx, shot._sx, dR0x, dϵx, dc0x, shot._dcx, shot._dsx)
end

function dZ_dρ(shot::Shot, ρ::Real, θ::Real)
    k, nu_ou, nu_eu, nu_ol, nu_el, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el = compute_both_bases(shot.ρ, ρ)
    R0x = evaluate_inbounds(shot.R0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ϵx = evaluate_inbounds(shot.ϵfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    κx = evaluate_inbounds(shot.κfe, k, nu_ou, nu_eu, nu_ol, nu_el)

    dR0x = evaluate_inbounds(shot.R0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dZ0x = evaluate_inbounds(shot.Z0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dϵx = evaluate_inbounds(shot.ϵfe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dκx = evaluate_inbounds(shot.κfe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)

    return MillerExtendedHarmonic.dZ_dρ(θ, R0x, ϵx, κx, dR0x, dZ0x, dϵx, dκx)
end

function dR_dθ(shot::Shot, ρ::Real, θ::Real)
    k, nu_ou, nu_eu, nu_ol, nu_el = compute_bases(shot.ρ, ρ)
    R0x = evaluate_inbounds(shot.R0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ϵx = evaluate_inbounds(shot.ϵfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    c0x = evaluate_inbounds(shot.c0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    evaluate_csx!(shot, k, nu_ou, nu_eu, nu_ol, nu_el)
    return MillerExtendedHarmonic.dR_dθ(θ, R0x, ϵx, c0x, shot._cx, shot._sx)
end

function dZ_dθ(shot::Shot, ρ::Real, θ::Real)
    k, nu_ou, nu_eu, nu_ol, nu_el = compute_bases(shot.ρ, ρ)
    R0x = evaluate_inbounds(shot.R0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ϵx = evaluate_inbounds(shot.ϵfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    κx = evaluate_inbounds(shot.κfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    return MillerExtendedHarmonic.dZ_dθ(θ, R0x, ϵx, κx)
end

function Jacobian(shot::Shot, ρ::Real, θ::Real)
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

    return MillerExtendedHarmonic.Jacobian(θ, R0x, ϵx, κx, c0x, shot._cx, shot._sx, dR0x, dZ0x, dϵx, dκx, dc0x, shot._dcx, shot._dsx)
end

function JacMat(shot::Shot, ρ::Real, θ::Real)
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

    return MillerExtendedHarmonic.JacMat(θ, R0x, ϵx, κx, c0x, shot._cx, shot._sx, dR0x, dZ0x, dϵx, dκx, dc0x, shot._dcx, shot._dsx)
end

function ∇ρ(shot::Shot, ρ::Real, θ::Real)
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

    return MillerExtendedHarmonic.∇ρ(θ, R0x, ϵx, κx, c0x, shot._cx, shot._sx, dR0x, dZ0x, dϵx, dκx, dc0x, shot._dcx, shot._dsx)
end

function ∇ρ2(shot::Shot, ρ::Real, θ::Real)
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

    return MillerExtendedHarmonic.∇ρ2(θ, R0x, ϵx, κx, c0x, shot._cx, shot._sx, dR0x, dZ0x, dϵx, dκx, dc0x, shot._dcx, shot._dsx)
end

function ∇θ(shot::Shot, ρ::Real, θ::Real)
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

    return MillerExtendedHarmonic.∇θ(θ, R0x, ϵx, κx, c0x, shot._cx, shot._sx, dR0x, dZ0x, dϵx, dκx, dc0x, shot._dcx, shot._dsx)
end

function ∇θ2(shot::Shot, ρ::Real, θ::Real)
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

    return MillerExtendedHarmonic.∇θ2(θ, R0x, ϵx, κx, c0x, shot._cx, shot._sx, dR0x, dZ0x, dϵx, dκx, dc0x, shot._dcx, shot._dsx)
end

function gρρ(shot::Shot, ρ::Real, θ::Real)
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

    return MillerExtendedHarmonic.gρρ(θ, R0x, ϵx, κx, c0x, shot._cx, shot._sx, dR0x, dZ0x, dϵx, dκx, dc0x, shot._dcx, shot._dsx)
end

function gρθ(shot::Shot, ρ::Real, θ::Real)
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

    return MillerExtendedHarmonic.gρθ(θ, R0x, ϵx, κx, c0x, shot._cx, shot._sx, dR0x, dZ0x, dϵx, dκx, dc0x, shot._dcx, shot._dsx)
end

function gθθ(shot::Shot, ρ::Real, θ::Real)
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

    return MillerExtendedHarmonic.gθθ(θ, R0x, ϵx, κx, c0x, shot._cx, shot._sx, dR0x, dZ0x, dϵx, dκx, dc0x, shot._dcx, shot._dsx)
end

function gρρ_gρθ(shot::Shot, ρ::Real, θ::Real)
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

    return MillerExtendedHarmonic.gρρ_gρθ(θ, R0x, ϵx, κx, c0x, shot._cx, shot._sx, dR0x, dZ0x, dϵx, dκx, dc0x, shot._dcx, shot._dsx)
end

function gρθ_gθθ(shot::Shot, ρ::Real, θ::Real)
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

    return MillerExtendedHarmonic.gρθ_gθθ(θ, R0x, ϵx, κx, c0x, shot._cx, shot._sx, dR0x, dZ0x, dϵx, dκx, dc0x, shot._dcx, shot._dsx)
end

function gρρ_gρθ_gθθ(shot::Shot, ρ::Real, θ::Real)
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

    return MillerExtendedHarmonic.gρρ_gρθ_gθθ(θ, R0x, ϵx, κx, c0x, shot._cx, shot._sx, dR0x, dZ0x, dϵx, dκx, dc0x, shot._dcx, shot._dsx)
end