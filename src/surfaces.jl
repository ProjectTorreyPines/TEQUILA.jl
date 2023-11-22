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

function concentric_surface!(surface::AbstractVector{<:Real}, x::Real, boundary::AbstractVector{<:Real}; Raxis::Real = boundary[1], Zaxis::Real = boundary[2])
    surface .= boundary
    return scale_surface!(surface, x; Raxis, Zaxis)
end

function scale_surface!(surface::AbstractVector{<:Real}, x::Real; Raxis::Real = surface[1], Zaxis::Real = surface[2])
    # these go to zero as you go to axis
    a = surface[1] * surface[3] * x
    surface[1] = x * surface[1] + (1.0 - x) * Raxis
    surface[2] = x * surface[2] + (1.0 - x) * Zaxis
    surface[3]  = a / surface[1]
    if x < 1.0
        surface[6:end] .*= x
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

function surfaces_FE(ρ:: AbstractVector{<:Real}, surfaces:: AbstractMatrix{<:Real}, flat_δ2::Nothing=nothing, flat_δ3::Nothing=nothing)

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

function surfaces_FE(ρ:: AbstractVector{<:Real}, surfaces:: AbstractMatrix{<:Real}, flat_δ2::AbstractVector{<:Real}, flat_δ3::AbstractVector{<:Real})

    R0fe, Z0fe, ϵfe, κfe, c0fe, cfe, sfe = surfaces_FE(ρ, surfaces)

    ρ1 = ρ[end-1]
    ρ4 = ρ[end]
    δρ = ρ4 - ρ1

    ρ2 = ρ1 + δ_frac_2 * δρ
    ρ3 = ρ1 + δ_frac_3 * δρ
    _, nu_ou2, nu_eu2, nu_ol2, nu_el2 = compute_bases(ρ, ρ2)
    _, nu_ou3, nu_eu3, nu_ol3, nu_el3 = compute_bases(ρ, ρ3)
    nus = (nu_ou2, nu_eu2, nu_ol2, nu_el2, nu_ou3, nu_eu3, nu_ol3, nu_el3)

    update_edge_derivatives!(R0fe, flat_δ2[1], flat_δ3[1], nus...)
    update_edge_derivatives!(Z0fe, flat_δ2[2], flat_δ3[2], nus...)
    update_edge_derivatives!(ϵfe,  flat_δ2[3], flat_δ3[3], nus...)
    update_edge_derivatives!(κfe,  flat_δ2[4], flat_δ3[4], nus...)
    update_edge_derivatives!(c0fe, flat_δ2[5], flat_δ3[5], nus...)
    M = length(cfe)
    for m in eachindex(cfe)
        update_edge_derivatives!(cfe[m], flat_δ2[5+m],   flat_δ3[5+m],   nus...)
        update_edge_derivatives!(sfe[m], flat_δ2[5+m+M], flat_δ3[5+m+M], nus...)
    end

    return R0fe, Z0fe, ϵfe, κfe, c0fe, cfe, sfe
end

function update_edge_derivatives!(Yfe, Y2, Y3, nu_ou2, nu_eu2, nu_ol2, nu_el2, nu_ou3, nu_eu3, nu_ol3, nu_el3)

    # deterimant of |nu_ou2   nu_ol2|
    #               |nu_ou3   nu_ol3|
    D = nu_ou2 * nu_ol3 - nu_ou3 * nu_ol2
    b2 = Y2 - Yfe.coeffs[end-2] * nu_eu2 - Yfe.coeffs[end] * nu_el2
    b3 = Y3 - Yfe.coeffs[end-2] * nu_eu3 - Yfe.coeffs[end] * nu_el3
    Yfe.coeffs[end-3] = (b2 * nu_ol3 - b3 * nu_ol2) / D
    Yfe.coeffs[end-1] = (nu_ou2 * b3 - nu_ou3 * b2) / D
end



function θ_at_RZ(shot::Shot, ρ::Real, R::Real, Z::Real; tid::Int = Threads.threadid())
    R0x, Z0x, ϵx, κx, c0x, cx, sx = compute_MXH(shot, ρ; tid)
    ax = R0x * ϵx
    bx = κx * ax
    return θ_at_RZ(R, Z, R0x, Z0x, ax, bx, c0x, cx, sx)
end

function θ_at_RZ(R::Real, Z::Real, R0x::Real, Z0x::Real, ax::Real, bx::Real, c0x::Real,
                 cx::AbstractVector{<:Real}, sx::AbstractVector{<:Real})
    θ = zero(promote_type(typeof(Z0x), typeof(Z), typeof(bx)))
    if bx != 0.0
        aa = min(1.0, max(-1.0, (Z0x - Z) / bx))
        θ = asin(aa)
    end
    signθ = (θ < 0.0) ? -1.0 : 1.0
    minmax = (signθ > 0.0) ? :min : :max
    R_at_Zext = MillerExtendedHarmonic.R_at_Zext(minmax, R0x, c0x, cx, sx, ax)
    (R < R_at_Zext) && (θ = signθ * π - θ)
    return θ, R_at_Zext
end

function Δ(shot, ρ, R, Z; tid = Threads.threadid())
    R0x, Z0x, ϵx, κx, c0x, cx, sx = compute_MXH(shot, ρ; tid)
    ax = R0x * ϵx
    bx = κx * ax

    θ, R_at_Zext = θ_at_RZ(R, Z, R0x, Z0x, ax, bx, c0x, cx, sx)

    Rx = MillerExtendedHarmonic.R_MXH(θ, R0x, c0x, cx, sx, ax)
    Zx = MillerExtendedHarmonic.Z_MXH(θ, Z0x, κx, ax)
    Δ = R < R_at_Zext ? R - Rx : Rx - R
    Δ -= abs(Z-Zx)
    return Δ
end

function ρθ_RZ(shot, R, Z)
    f = x -> Δ(shot, x, R, Z)
    ρ = f(1.0) < 0.0 ? 1.0 : Roots.find_zero(f, (0,1), Roots.A42())
    θ, _ = (ρ == 0.0) ? (0.0, 0.0)  : θ_at_RZ(shot, ρ, R, Z)
    return ρ, θ
end

##########################################################
# BCL 8/24/22: THESE SHOULD ALL MAKE USE OF compute_bases
##########################################################

function evaluate_csx!(shot::Shot, k::Integer, nu_ou, nu_eu, nu_ol, nu_el; tid = Threads.threadid())
    return evaluate_csx!(shot._cx, shot._sx, shot.cfe, shot.sfe, k, nu_ou, nu_eu, nu_ol, nu_el; tid)
end

function evaluate_csx!(cxs, sxs, cfe, sfe, k::Integer, nu_ou, nu_eu, nu_ol, nu_el; tid = Threads.threadid())
    cx = get_tmp(cxs[tid], nu_ou)
    sx = get_tmp(sxs[tid], nu_ou)
    @inbounds for m in eachindex(cfe)
        cx[m] = evaluate_inbounds(cfe[m], k, nu_ou, nu_eu, nu_ol, nu_el)
        sx[m] = evaluate_inbounds(sfe[m], k, nu_ou, nu_eu, nu_ol, nu_el)
    end
    return cx, sx
end

function evaluate_csx(cfe, sfe, k::Integer, nu_ou, nu_eu, nu_ol, nu_el)
    cx = [evaluate_inbounds(c, k, nu_ou, nu_eu, nu_ol, nu_el) for c in cfe]
    sx = [evaluate_inbounds(s, k, nu_ou, nu_eu, nu_ol, nu_el) for s in sfe]
    return cx, sx
end

function evaluate_dcsx!(shot::Shot, k::Integer, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el; tid = Threads.threadid())
    return evaluate_dcsx!(shot._dcx, shot._dsx, shot.cfe, shot.sfe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el; tid)
end

function evaluate_dcsx!(dcxs, dsxs, cfe, sfe, k::Integer, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el; tid = Threads.threadid())
    dcx = get_tmp(dcxs[tid], D_nu_ou)
    dsx = get_tmp(dsxs[tid], D_nu_ou)
    @inbounds for m in eachindex(cfe)
        dcx[m] =  evaluate_inbounds(cfe[m], k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
        dsx[m] =  evaluate_inbounds(sfe[m], k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    end
    return dcx, dsx
end

function evaluate_dcsx(cfe, sfe, k::Integer, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dcx = [evaluate_inbounds(c, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el) for c in cfe]
    dsx = [evaluate_inbounds(s, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el) for s in sfe]
    return dcx, dsx
end

function compute_MXH(shot::Shot, ρ::Real; tid = Threads.threadid())
    return compute_MXH(shot.ρ, ρ, shot.R0fe, shot.Z0fe, shot.ϵfe, shot.κfe, shot.c0fe,
                       shot.cfe, shot.sfe, shot._cx, shot._sx; tid)
end

function compute_MXH(ρs::AbstractVector{<:Real}, ρ::Real, R0fe, Z0fe, ϵfe, κfe, c0fe, cfe, sfe, cxs, sxs; tid = Threads.threadid())
    k, nu_ou, nu_eu, nu_ol, nu_el = compute_bases(ρs, ρ)
    R0x = evaluate_inbounds(R0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    Z0x = evaluate_inbounds(Z0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ϵx = evaluate_inbounds(ϵfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    κx = evaluate_inbounds(κfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    c0x = evaluate_inbounds(c0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    cx, sx = evaluate_csx!(cxs, sxs, cfe, sfe, k, nu_ou, nu_eu, nu_ol, nu_el; tid)
    return R0x, Z0x, ϵx, κx, c0x, cx, sx
end

function compute_MXH(ρs::AbstractVector{<:Real}, ρ::Real, R0fe, Z0fe, ϵfe, κfe, c0fe, cfe, sfe)
    k, nu_ou, nu_eu, nu_ol, nu_el = compute_bases(ρs, ρ)
    R0x = evaluate_inbounds(R0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    Z0x = evaluate_inbounds(Z0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ϵx = evaluate_inbounds(ϵfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    κx = evaluate_inbounds(κfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    c0x = evaluate_inbounds(c0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    cx, sx = evaluate_csx(cfe, sfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    return R0x, Z0x, ϵx, κx, c0x, cx, sx
end

function compute_D_MXH(shot::Shot, ρ::Real; tid = Threads.threadid())
    return compute_D_MXH(shot.ρ, ρ, shot.R0fe, shot.Z0fe, shot.ϵfe, shot.κfe, shot.c0fe,
                       shot.cfe, shot.sfe, shot._dcx, shot._dsx; tid)
end

function compute_D_MXH(ρs::AbstractVector{<:Real}, ρ::Real, R0fe, Z0fe, ϵfe, κfe, c0fe, cfe, sfe, dcxs, dsxs; tid = Threads.threadid())
    k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el = compute_D_bases(ρs, ρ)
    dR0x = evaluate_inbounds(R0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dZ0x = evaluate_inbounds(Z0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dϵx = evaluate_inbounds(ϵfe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dκx = evaluate_inbounds(κfe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dc0x = evaluate_inbounds(c0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dcx, dsx = evaluate_dcsx!(dcxs, dsxs, cfe, sfe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el; tid)
    return dR0x, dZ0x, dϵx, dκx, dc0x, dcx, dsx
end

function compute_D_MXH(ρs::AbstractVector{<:Real}, ρ::Real, R0fe, Z0fe, ϵfe, κfe, c0fe, cfe, sfe)
    k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el = compute_D_bases(ρs, ρ)
    dR0x = evaluate_inbounds(R0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dZ0x = evaluate_inbounds(Z0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dϵx = evaluate_inbounds(ϵfe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dκx = evaluate_inbounds(κfe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dc0x = evaluate_inbounds(c0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dcx, dsx = evaluate_dcsx(cfe, sfe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    return dR0x, dZ0x, dϵx, dκx, dc0x, dcx, dsx
end

function compute_both_MXH(shot::Shot, ρ::Real; tid = Threads.threadid())
    return compute_both_MXH(shot.ρ, ρ, shot.R0fe, shot.Z0fe, shot.ϵfe, shot.κfe, shot.c0fe,
                       shot.cfe, shot.sfe, shot._cx, shot._sx, shot._dcx, shot._dsx; tid)
end

function compute_both_MXH(ρs::AbstractVector{<:Real}, ρ::Real, R0fe, Z0fe, ϵfe, κfe, c0fe, cfe, sfe, cxs, sxs, dcxs, dsxs; tid = Threads.threadid())
    k, nu_ou, nu_eu, nu_ol, nu_el, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el = compute_both_bases(ρs, ρ)
    R0x = evaluate_inbounds(R0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    Z0x = evaluate_inbounds(Z0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ϵx = evaluate_inbounds(ϵfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    κx = evaluate_inbounds(κfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    c0x = evaluate_inbounds(c0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    cx, sx = evaluate_csx!(cxs, sxs, cfe, sfe, k, nu_ou, nu_eu, nu_ol, nu_el; tid)
    dR0x = evaluate_inbounds(R0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dZ0x = evaluate_inbounds(Z0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dϵx = evaluate_inbounds(ϵfe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dκx = evaluate_inbounds(κfe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dc0x = evaluate_inbounds(c0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dcx, dsx = evaluate_dcsx!(dcxs, dsxs, cfe, sfe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el; tid)
    return R0x, Z0x, ϵx, κx, c0x, cx, sx, dR0x, dZ0x, dϵx, dκx, dc0x, dcx, dsx
end

function compute_both_MXH(ρs::AbstractVector{<:Real}, ρ::Real, R0fe, Z0fe, ϵfe, κfe, c0fe, cfe, sfe)
    k, nu_ou, nu_eu, nu_ol, nu_el, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el = compute_both_bases(ρs, ρ)
    R0x = evaluate_inbounds(R0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    Z0x = evaluate_inbounds(Z0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ϵx = evaluate_inbounds(ϵfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    κx = evaluate_inbounds(κfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    c0x = evaluate_inbounds(c0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    cx, sx = evaluate_csx(cfe, sfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    dR0x = evaluate_inbounds(R0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dZ0x = evaluate_inbounds(Z0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dϵx = evaluate_inbounds(ϵfe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dκx = evaluate_inbounds(κfe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dc0x = evaluate_inbounds(c0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dcx, dsx = evaluate_dcsx(cfe, sfe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    return R0x, Z0x, ϵx, κx, c0x, cx, sx, dR0x, dZ0x, dϵx, dκx, dc0x, dcx, dsx
end



function MillerExtendedHarmonic.MXH(shot::Shot, ρ::Real; tid = Threads.threadid())
    R0x, Z0x, ϵx, κx, c0x, cx, sx = compute_MXH(shot, ρ; tid)
    return MXH(R0x, Z0x, ϵx, κx, c0x, cx, sx)
end


function Tr(shot::Shot, ρ::Real, θ::Real; tid = Threads.threadid())
    k, nu_ou, nu_eu, nu_ol, nu_el = compute_bases(shot.ρ, ρ)
    c0 = evaluate_inbounds(shot.c0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    cx, sx = evaluate_csx!(shot, k, nu_ou, nu_eu, nu_ol, nu_el; tid)
    return MillerExtendedHarmonic.Tr(θ, c0, cx, sx)
end

function dTr_dρ(shot::Shot, ρ::Real, θ::Real; tid = Threads.threadid())
    k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el = compute_D_bases(shot.ρ, ρ)
    dc0 = evaluate_inbounds(shot.c0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dcx, dsx = evaluate_dcsx!(shot, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el; tid)
    return MillerExtendedHarmonic.dTr_dρ(θ, dc0, dcx, dsx) # ρ derivative just passes through
end

function dTr_dθ(shot::Shot, ρ::Real, θ::Real; tid = Threads.threadid())
    k, nu_ou, nu_eu, nu_ol, nu_el = compute_bases(shot.ρ, ρ)
    cx, sx = evaluate_csx!(shot, k, nu_ou, nu_eu, nu_ol, nu_el; tid)
    return MillerExtendedHarmonic.dTr_dθ(θ, cx, sx)
end

function R(shot::Shot, ρ::Real, θ::Real; tid = Threads.threadid())
    k, nu_ou, nu_eu, nu_ol, nu_el = compute_bases(shot.ρ, ρ)
    R0x = evaluate_inbounds(shot.R0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ϵx = evaluate_inbounds(shot.ϵfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    c0x = evaluate_inbounds(shot.c0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    cx, sx = evaluate_csx!(shot, k, nu_ou, nu_eu, nu_ol, nu_el; tid)
    ax = R0x * ϵx

    return MillerExtendedHarmonic.R_MXH(θ, R0x, c0x, cx, sx, ax)
end

function Rmin(shot::Shot, ρ::Real)
    k, nu_ou, nu_eu, nu_ol, nu_el = compute_bases(shot.ρ, ρ)
    R0x = evaluate_inbounds(shot.R0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ϵx = evaluate_inbounds(shot.ϵfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ax = R0x * ϵx
    return R0x - ax
end

function Rmax(shot::Shot, ρ::Real)
    k, nu_ou, nu_eu, nu_ol, nu_el = compute_bases(shot.ρ, ρ)
    R0x = evaluate_inbounds(shot.R0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ϵx = evaluate_inbounds(shot.ϵfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ax = R0x * ϵx
    return R0x + ax
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

function Zmin(shot::Shot, ρ::Real)
    k, nu_ou, nu_eu, nu_ol, nu_el = compute_bases(shot.ρ, ρ)
    R0x = evaluate_inbounds(shot.R0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    Z0x = evaluate_inbounds(shot.Z0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ϵx = evaluate_inbounds(shot.ϵfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    κx = evaluate_inbounds(shot.κfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ax = R0x * ϵx
    bx = ax * κx
    return Z0x - bx
end

function Zmax(shot::Shot, ρ::Real)
    k, nu_ou, nu_eu, nu_ol, nu_el = compute_bases(shot.ρ, ρ)
    R0x = evaluate_inbounds(shot.R0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    Z0x = evaluate_inbounds(shot.Z0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ϵx = evaluate_inbounds(shot.ϵfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    κx = evaluate_inbounds(shot.κfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ax = R0x * ϵx
    bx = ax * κx
    return Z0x + bx
end

function R_Z(shot::Shot, ρ::Real, θ::Real; tid = Threads.threadid())
    R0x, Z0x, ϵx, κx, c0x, cx, sx = compute_MXH(shot, ρ; tid)
    ax = R0x * ϵx
    R = MillerExtendedHarmonic.R_MXH(θ, R0x, c0x, cx, sx, ax)
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

function dR_dρ(shot::Shot, ρ::Real, θ::Real; tid = Threads.threadid())
    k, nu_ou, nu_eu, nu_ol, nu_el, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el = compute_both_bases(shot.ρ, ρ)
    R0x = evaluate_inbounds(shot.R0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ϵx = evaluate_inbounds(shot.ϵfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    c0x = evaluate_inbounds(shot.c0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    cx, sx = evaluate_csx!(shot, k, nu_ou, nu_eu, nu_ol, nu_el; tid)

    dR0x = evaluate_inbounds(shot.R0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dϵx = evaluate_inbounds(shot.ϵfe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dc0x = evaluate_inbounds(shot.c0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dcx, dsx = evaluate_dcsx!(shot, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el; tid)

    return MillerExtendedHarmonic.dR_dρ(θ, R0x, ϵx, c0x, cx, sx, dR0x, dϵx, dc0x, dcx, dsx)
end

function dZ_dρ(shot::Shot, ρ::Real, θ::Real; tid = Threads.threadid())
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

function dR_dθ(shot::Shot, ρ::Real, θ::Real; tid = Threads.threadid())
    k, nu_ou, nu_eu, nu_ol, nu_el = compute_bases(shot.ρ, ρ)
    R0x = evaluate_inbounds(shot.R0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ϵx = evaluate_inbounds(shot.ϵfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    c0x = evaluate_inbounds(shot.c0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    cx, sx = evaluate_csx!(shot, k, nu_ou, nu_eu, nu_ol, nu_el; tid)
    return MillerExtendedHarmonic.dR_dθ(θ, R0x, ϵx, c0x, cx, sx)
end

function dZ_dθ(shot::Shot, ρ::Real, θ::Real)
    k, nu_ou, nu_eu, nu_ol, nu_el = compute_bases(shot.ρ, ρ)
    R0x = evaluate_inbounds(shot.R0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ϵx = evaluate_inbounds(shot.ϵfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    κx = evaluate_inbounds(shot.κfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    return MillerExtendedHarmonic.dZ_dθ(θ, R0x, ϵx, κx)
end

function Jacobian(shot::Shot, ρ::Real, θ::Real; tid = Threads.threadid())
    R0x, _, ϵx, κx, c0x, cx, sx, dR0x, dZ0x, dϵx, dκx, dc0x, dcx, dsx = compute_both_MXH(shot, ρ; tid)
    return MillerExtendedHarmonic.Jacobian(θ, R0x, ϵx, κx, c0x, cx, sx, dR0x, dZ0x, dϵx, dκx, dc0x, dcx, dsx)
end

function JacMat(shot::Shot, ρ::Real, θ::Real; tid = Threads.threadid())
    R0x, _, ϵx, κx, c0x, cx, sx, dR0x, dZ0x, dϵx, dκx, dc0x, dcx, dsx = compute_both_MXH(shot, ρ; tid)
    R_ρ, R_θ, Z_ρ, Z_θ = MillerExtendedHarmonic.JacMat(θ, R0x, ϵx, κx, c0x, cx, sx, dR0x, dZ0x, dϵx, dκx, dc0x, dcx, dsx)
    return @SMatrix[R_ρ R_θ; Z_ρ Z_θ]
end

function ∇ρ(shot::Shot, ρ::Real, θ::Real; tid = Threads.threadid())
    R0x, _, ϵx, κx, c0x, cx, sx, dR0x, dZ0x, dϵx, dκx, dc0x, dcx, dsx = compute_both_MXH(shot, ρ; tid)
    return MillerExtendedHarmonic.∇ρ(θ, R0x, ϵx, κx, c0x, cx, sx, dR0x, dZ0x, dϵx, dκx, dc0x, dcx, dsx)
end

function ∇ρ2(shot::Shot, ρ::Real, θ::Real; tid = Threads.threadid())
    R0x, _, ϵx, κx, c0x, cx, sx, dR0x, dZ0x, dϵx, dκx, dc0x, dcx, dsx = compute_both_MXH(shot, ρ; tid)
    return MillerExtendedHarmonic.∇ρ2(θ, R0x, ϵx, κx, c0x, cx, sx, dR0x, dZ0x, dϵx, dκx, dc0x, dcx, dsx)
end

function ∇θ(shot::Shot, ρ::Real, θ::Real; tid = Threads.threadid())
    R0x, _, ϵx, κx, c0x, cx, sx, dR0x, dZ0x, dϵx, dκx, dc0x, dcx, dsx = compute_both_MXH(shot, ρ; tid)
    return MillerExtendedHarmonic.∇θ(θ, R0x, ϵx, κx, c0x, cx, sx, dR0x, dZ0x, dϵx, dκx, dc0x, dcx, dsx)
end

function ∇θ2(shot::Shot, ρ::Real, θ::Real; tid = Threads.threadid())
    R0x, _, ϵx, κx, c0x, cx, sx, dR0x, dZ0x, dϵx, dκx, dc0x, dcx, dsx = compute_both_MXH(shot, ρ; tid)
    return MillerExtendedHarmonic.∇θ2(θ, R0x, ϵx, κx, c0x, cx, sx, dR0x, dZ0x, dϵx, dκx, dc0x, dcx, dsx)
end

function gρρ(shot::Shot, ρ::Real, θ::Real; tid = Threads.threadid())
    R0x, _, ϵx, κx, c0x, cx, sx, dR0x, dZ0x, dϵx, dκx, dc0x, dcx, dsx = compute_both_MXH(shot, ρ; tid)
    return MillerExtendedHarmonic.gρρ(θ, R0x, ϵx, κx, c0x, cx, sx, dR0x, dZ0x, dϵx, dκx, dc0x, dcx, dsx)
end

function gρθ(shot::Shot, ρ::Real, θ::Real; tid = Threads.threadid())
    R0x, _, ϵx, κx, c0x, cx, sx, dR0x, dZ0x, dϵx, dκx, dc0x, dcx, dsx = compute_both_MXH(shot, ρ; tid)
    return MillerExtendedHarmonic.gρθ(θ, R0x, ϵx, κx, c0x, cx, sx, dR0x, dZ0x, dϵx, dκx, dc0x, dcx, dsx)
end

function gθθ(shot::Shot, ρ::Real, θ::Real; tid = Threads.threadid())
    R0x, _, ϵx, κx, c0x, cx, sx, dR0x, dZ0x, dϵx, dκx, dc0x, dcx, dsx = compute_both_MXH(shot, ρ; tid)
    return MillerExtendedHarmonic.gθθ(θ, R0x, ϵx, κx, c0x, cx, sx, dR0x, dZ0x, dϵx, dκx, dc0x, dcx, dsx)
end

function gρρ_gρθ(shot::Shot, ρ::Real, θ::Real; tid = Threads.threadid())
    R0x, _, ϵx, κx, c0x, cx, sx, dR0x, dZ0x, dϵx, dκx, dc0x, dcx, dsx = compute_both_MXH(shot, ρ; tid)
    return MillerExtendedHarmonic.gρρ_gρθ(θ, R0x, ϵx, κx, c0x, cx, sx, dR0x, dZ0x, dϵx, dκx, dc0x, dcx, dsx)
end

function gρθ_gθθ(shot::Shot, ρ::Real, θ::Real; tid = Threads.threadid())
    R0x, _, ϵx, κx, c0x, cx, sx, dR0x, dZ0x, dϵx, dκx, dc0x, dcx, dsx = compute_both_MXH(shot, ρ; tid)
    return MillerExtendedHarmonic.gρθ_gθθ(θ, R0x, ϵx, κx, c0x, cx, sx, dR0x, dZ0x, dϵx, dκx, dc0x, dcx, dsx)
end

function gρρ_gρθ_gθθ(shot::Shot, ρ::Real, θ::Real; tid = Threads.threadid())
    R0x, _, ϵx, κx, c0x, cx, sx, dR0x, dZ0x, dϵx, dκx, dc0x, dcx, dsx = compute_both_MXH(shot, ρ; tid)
    return MillerExtendedHarmonic.gρρ_gρθ_gθθ(θ, R0x, ϵx, κx, c0x, cx, sx, dR0x, dZ0x, dϵx, dκx, dc0x, dcx, dsx, shot.Q.Fsin, shot.Q.Fcos)
end