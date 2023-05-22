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

outside(S::Union{MXH, AbstractVector{<:Real}}, x) = !in_surface(x[1], x[2], S);

function surface_bracket(shot::Shot, R::Real, Z::Real)

    x = SVector(R, Z)

    _, Ncol = size(shot.surfaces)

    get_col(k::Integer) = @views shot.surfaces[:, k]
    get_col(x::SVector{2,<:Real}) = x

    ko = searchsortedfirst(1:Ncol, x, by=get_col, lt=outside)

    ko == 1 && return 1, shot.ρ[1], 0.0, 1, shot.ρ[1], 0.0

    if ko > shot.N
        ρ = shot.ρ[end]
        @views θ = nearest_angle(R, Z, shot.surfaces[:, end])
        return shot.N, ρ, θ, shot.N, ρ, θ
    end

    ρo = shot.ρ[ko]
    @views So = shot.surfaces[:, ko]
    θo = nearest_angle(R, Z, So)
    Ro = R_MXH(θo, So)
    Zo = Z_MXH(θo, So)
    (R==Ro && Z==Zo) && return ko, ρo, θo, ko, ρo, θo

    ki = ko - 1
    ρi = shot.ρ[ki]
    @views θi = nearest_angle(R, Z, shot.surfaces[:, ki])
    return ki, ρi, θi, ko, ρo, θo
end

function θr_oi(θo, θi, c0, cfe, sfe, k, nu_ou, nu_eu, nu_ol, nu_el)

    θro = θo + c0
    θri = θi + c0
    @inbounds for m in eachindex(cfe)
        C = evaluate_inbounds(cfe[m], k, nu_ou, nu_eu, nu_ol, nu_el)
        S = evaluate_inbounds(sfe[m], k, nu_ou, nu_eu, nu_ol, nu_el)
        scm = sincos(m * θo)
        θro += S * scm[1] + C * scm[2]
        scm = sincos(m * θi)
        θri += S * scm[1] + C * scm[2]
    end
    return θro, θri
end

function res_find_ρ(ρ::Real, shot::Shot, R::Real, Z::Real; return_θ=false)

    k, nu_ou, nu_eu, nu_ol, nu_el = compute_bases(shot.R0fe.x, ρ)
    R0 = evaluate_inbounds(shot.R0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    Z0 = evaluate_inbounds(shot.Z0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ϵ = evaluate_inbounds(shot.ϵfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    κ = evaluate_inbounds(shot.κfe, k, nu_ou, nu_eu, nu_ol, nu_el)

    a = R0 * ϵ
    b = κ * a

    dZ = (Z0 - Z)
    aa = dZ / b
    aa = min(1.0, max(-1.0, aa))
    θo = asin(aa)
    signθ = θo < 0.0 ? -1.0 : 1.0
    θi = signθ * π - θo

    c0 = evaluate_inbounds(shot.c0fe, k, nu_ou, nu_eu, nu_ol, nu_el)

    θro, θri = θr_oi(θo, θi, c0, shot.cfe, shot.sfe, k, nu_ou, nu_eu, nu_ol, nu_el)

    Ro = R0 + a * cos(θro)
    Ri = R0 + a * cos(θri)

    reso =  Ro - R
    resi =  Ri - R
    sign_res = sign(reso) * sign(resi)
    if abs(reso) < abs(resi)
        return return_θ ? θo : sign_res * abs(reso)
    end
    return return_θ ? θi : sign_res * abs(resi)
end

function res_zext(ρ::Real, R0fe::FE_rep, Z0fe::FE_rep, ϵfe::FE_rep, κfe::FE_rep, Z::Real)
    k, nu_ou, nu_eu, nu_ol, nu_el = compute_bases(R0fe.x, ρ)
    R0 = evaluate_inbounds(R0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    Z0 = evaluate_inbounds(Z0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ϵ = evaluate_inbounds(ϵfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    κ = evaluate_inbounds(κfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    b = R0 * ϵ * κ
    return Z0 + sign(Z-Z0) * b - Z
end

function res_zext(ρ::Real, shot::Shot, Z::Real; debug=false)
    k, nu_ou, nu_eu, nu_ol, nu_el = compute_bases(shot.R0fe.x, ρ)
    R0 = evaluate_inbounds(shot.R0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    Z0 = evaluate_inbounds(shot.Z0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ϵ = evaluate_inbounds(shot.ϵfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    κ = evaluate_inbounds(shot.κfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    b = R0 * ϵ * κ
    sign_dZ = (Z <= Z0) ? -1 : 1
    debug && println(Z0)
    return Z0 + sign_dZ * b - Z
end

function ρθ_approx(shot::Shot, R::Real, Z::Real)

    ki, ρi, θi, ko, ρo, θo = surface_bracket(shot, R, Z)
    ki==ko && return ρo, θo # on a surface exactly

    @views So = shot.surfaces[:, ko]
    Ro = R_MXH(θo, So)
    Zo = Z_MXH(θo, So)

    Δo = sqrt((R - Ro)^2 + (Z - Zo)^2)

    @views Si = shot.surfaces[:, ki]
    Ri = R_MXH(θi, Si)
    Zi = Z_MXH(θi, Si)
    Δi = sqrt((R - Ri)^2 + (Z - Zi)^2)

    # linearly interpolate in Δ
    invΔ = 1.0 / (Δi + Δo)
    ρ = (ρo * Δi + ρi * Δo) * invΔ
    θ = (θo * Δi + θi * Δo) * invΔ
    return ρ, θ
end

ρθ_RZ(shot::Shot, R::Real, Z::Real) = ρθ_RZ4(shot, R, Z)

function ρθ_RZ1(shot::Shot, R::Real, Z::Real)

    ki, ρi, θi, ko, ρo, θo = surface_bracket(shot, R, Z)
    ki==ko && return ρo, θo # on a surface exactly
    if abs(θi) == 0.5 * π
        # find ρ where Z = Zext
        f_zext(x) = res_zext(x, shot, Z)^2
        ρi = optimize(f_zext, ρi, ρo).minimizer
        #ρi = Roots.secant_method(f_zext, (ρi, ρo))
    end
    f_find_ρ(x) = res_find_ρ(x, shot, R, Z)^2
    ρ = optimize(f_find_ρ, ρi, ρo).minimizer
    #ρ = Roots.secant_method(f_find_ρ, (ρi, ρo))
    # x0 = (ρi, ρo)
    # M = Secant()
    # ZP = ZeroProblem(f_find_ρ, x0)
    # ρ = solve(ZP, M)

    θ = res_find_ρ(ρ, shot, R, Z, return_θ=true)

    return ρ, θ
end

function ρθ_RZ3(shot::Shot, R::Real, Z::Real)

    f_zext(x) = res_zext(x, shot, Z)^2
    ρi = optimize(f_zext, 0, 1).minimizer
    f_find_ρ(x) = res_find_ρ(x, shot, R, Z)^2
    ρ = optimize(f_find_ρ, ρi, 1.0).minimizer
    θ = res_find_ρ(ρ, shot, R, Z, return_θ=true)
    return ρ, θ
end

function ρθ_RZ4(shot::Shot, R::Real, Z::Real)
    @views if !in_surface(R, Z, shot.surfaces[:,end])
        ρ = 1.0
        θ = res_find_ρ(ρ, shot, R, Z, return_θ=true)
        return ρ, θ
    else
        f_zext(x;debug=false) = res_zext(x, shot, Z;debug)
        try
            ρ = Roots.find_zero(f_zext, (0, 1), Roots.A42())
        catch err
            println((R, Z, f_zext(0.0;debug=true), f_zext(1.0;debug=true)))
            #println((f_find_ρ(0.0), f_find_ρ(1.0)))
            rethrow(err)
        end
        if ρ < 1.0
            f_find_ρ(x) = res_find_ρ(x, shot, R, Z)
            try
                ρ = Roots.find_zero(f_find_ρ, (ρ, 1.0), Roots.A42())
            catch err
                println((R, Z, ρ))
                println((f_find_ρ(ρ), f_find_ρ(1.0)))
                rethrow(err)
            end
        end
        θ = res_find_ρ(ρ, shot, R, Z, return_θ=true)
        return ρ, θ
    end
end

function ΔZext(shot, ρ, Z)
    tid = Threads.threadid()
    k, nu_ou, nu_eu, nu_ol, nu_el = TEQUILA.compute_bases(shot.ρ, ρ)
    R0x = TEQUILA.evaluate_inbounds(shot.R0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    Z0x = TEQUILA.evaluate_inbounds(shot.Z0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ϵx = TEQUILA.evaluate_inbounds(shot.ϵfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    κx = TEQUILA.evaluate_inbounds(shot.κfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ax = R0x * ϵx
    bx = κx * ax

    Δ = (Z < Z0x) ? Z - (Z0x - bx) : Z - (Z0x + bx)
    return Δ
end

function find_Zext(shot, Z)
    f(x) = ΔZext(shot, x, Z)
    return Roots.find_zero(f, (0,1), Roots.A42())
end

function Δ(shot, ρ, R, Z)
    tid = Threads.threadid()
    k, nu_ou, nu_eu, nu_ol, nu_el = TEQUILA.compute_bases(shot.ρ, ρ)
    R0x = TEQUILA.evaluate_inbounds(shot.R0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    Z0x = TEQUILA.evaluate_inbounds(shot.Z0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ϵx = TEQUILA.evaluate_inbounds(shot.ϵfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    κx = TEQUILA.evaluate_inbounds(shot.κfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    c0x = TEQUILA.evaluate_inbounds(shot.c0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    TEQUILA.evaluate_csx!(shot, k, nu_ou, nu_eu, nu_ol, nu_el; tid)
    
    ax = R0x * ϵx
    bx = κx * ax
    
    aa = min(1.0, max(-1.0, (Z0x - Z) / bx))
    θ = asin(aa)
    
    signθ = (θ < 0.0) ? -1.0 : 1.0
    θext = signθ * 0.5 * π
    Zext = Z0x - bx * sin(θext)
    R_at_Zext = MillerExtendedHarmonic.R_MXH(θext, R0x, c0x, shot._cx[tid], shot._sx[tid], ax)
    
    (R < R_at_Zext) && (θ = signθ * π - θ)
    
    Rx = MillerExtendedHarmonic.R_MXH(θ, R0x, c0x, shot._cx[tid], shot._sx[tid], ax)
    Δ = R - Rx

    Zx = MillerExtendedHarmonic.Z_MXH(θ, Z0x, κx, ax)
    Δ += sign(Δ) * abs(Z-Zx)
end

function ρθ_RZ2(shot, R, Z)
    f(x) = Δ(shot, x, R, Z)
    return Roots.find_zero(f, (0,1), Roots.A42())
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