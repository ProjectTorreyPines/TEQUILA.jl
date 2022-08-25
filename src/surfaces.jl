function concentric_surface(x::Real, boundary::MXH)
    surface = deepcopy(boundary)
    # these go to zero as you go to axis
    surface.ϵ *= x
    if x < 1.0
        surface.c .*= x
        surface.s .*= x
    end
    return surface
end

function concentric_surface!(surface::MXH, x::Real, boundary::MXH)
    copy_MXH!(surface, boundary)
    # these go to zero as you go to axis
    surface.ϵ *= x
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

function R_Z(shot::Shot, x::Real, t::Real)
    return R_Z(shot.R0fe, shot.Z0fe, shot.ϵfe, shot.κfe, shot.c0fe, shot.cfe, shot.sfe, x, t)
end

function R_Z(R0fe::FE_rep, Z0fe::FE_rep, ϵfe::FE_rep, κfe::FE_rep, c0fe::FE_rep,
             cfe::AbstractVector{<:FE_rep}, sfe::AbstractVector{<:FE_rep}, x::Real, t::Real)

    k, nu_ou, nu_eu, nu_ol, nu_el = compute_bases(R0fe.x, x)
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
        csx += dot((smx, cmx), sincos(m * t))
    end

    tr = t + c0x + csx
    R = R0x + ax * cos(tr)
    Z = Z0x - κx * ax * sin(t)
    return R, Z
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
        θro += dot((S, C), sincos(m * θo))
        θri += dot((S, C), sincos(m * θi))
    end
    return θro, θri
end

function res_find_ρ(ρ::Real, R0fe::FE_rep, Z0fe::FE_rep, ϵfe::FE_rep, κfe::FE_rep, c0fe::FE_rep,
                    cfe::AbstractVector{<:FE_rep}, sfe::AbstractVector{<:FE_rep}, R::Real, Z::Real; return_θ=false)

    k, nu_ou, nu_eu, nu_ol, nu_el = compute_bases(R0fe.x, ρ)
    R0 = evaluate_inbounds(R0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    Z0 = evaluate_inbounds(Z0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ϵ = evaluate_inbounds(ϵfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    κ = evaluate_inbounds(κfe, k, nu_ou, nu_eu, nu_ol, nu_el)

    a = R0 * ϵ
    b = κ * a

    dZ = (Z0 - Z)
    aa = dZ / b
    aa = min(1.0, max(-1.0, aa))
    θo = asin(aa)
    signθ = θo < 0.0 ? -1.0 : 1.0
    θi = signθ * π - θo

    c0 = evaluate_inbounds(c0fe, k, nu_ou, nu_eu, nu_ol, nu_el)

    θro, θri = θr_oi(θo, θi, c0, cfe, sfe, k, nu_ou, nu_eu, nu_ol, nu_el)

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

function ρθ_RZ(shot::Shot, R::Real, Z::Real)

    ki, ρi, θi, ko, ρo, θo = surface_bracket(shot, R, Z)
    ki==ko && return ρo, θo # on a surface exactly
    if abs(θi) == 0.5 * π
        # find ρ where Z = Zext

        f_zext(x) = res_zext(x, shot.R0fe, shot.Z0fe, shot.ϵfe, shot.κfe, Z)^2
        ρi = optimize(f_zext, ρi, ρo).minimizer
    end

    f_find_ρ(x) = res_find_ρ(x, shot.R0fe, shot.Z0fe, shot.ϵfe, shot.κfe, shot.c0fe, shot.cfe, shot.sfe, R, Z)^2
    ρ = optimize(f_find_ρ, ρi, ρo).minimizer

    θ = res_find_ρ(ρ, shot.R0fe, shot.Z0fe, shot.ϵfe, shot.κfe, shot.c0fe, shot.cfe, shot.sfe, R, Z, return_θ=true)

    return ρ, θ
end

##########################################################
# BCL 8/24/22: THESE SHOULD ALL MAKE USE OF compute_bases
##########################################################

Tr(ρ, θ, R0fe, Z0fe, ϵfe, κfe, c0fe, cfe, sfe) = Tr(ρ, θ, c0fe, cfe, sfe)
function Tr(ρ::Real, θ::Real, c0fe::FE_rep, cfe::AbstractVector{<:FE_rep}, sfe::AbstractVector{<:FE_rep})
    θr = θ + c0fe(ρ)
    @inbounds for m in eachindex(cfe)
        S = sfe[m](ρ)
        C = cfe[m](ρ)
        θr += dot((S, C), sincos(m * θ))
    end
    return θr
end

dTr_dρ(ρ, θ, R0fe, Z0fe, ϵfe, κfe, c0fe, cfe, sfe) = dTr_dρ(ρ, θ, c0fe, cfe, sfe)
function dTr_dρ(ρ::Real, θ::Real, c0fe::FE_rep, cfe::AbstractVector{<:FE_rep}, sfe::AbstractVector{<:FE_rep})
    dθr_dρ = D(c0fe, ρ)
    @inbounds for m in eachindex(cfe)
        dS = D(sfe[m], ρ)
        dC = D(cfe[m], ρ)
        dθr_dρ += dot((dS, dC), sincos(m * θ))
    end
    return dθr_dρ
end

dTr_dθ(ρ, θ, R0fe, Z0fe, ϵfe, κfe, c0fe, cfe, sfe) = dTr_dθ(ρ, θ, cfe, sfe)
function dTr_dθ(ρ::Real, θ::Real, cfe::AbstractVector{<:FE_rep}, sfe::AbstractVector{<:FE_rep})
    dθr_dθ = 1.0
    @inbounds for m in eachindex(cfe)
        S = sfe[m](ρ)
        C = cfe[m](ρ)
        dθr_dθ += dot((-C, S), sincos(m * θ))
    end
    return dθr_dθ
end

dR_dρ(ρ, θ, R0fe, Z0fe, ϵfe, κfe, c0fe, cfe, sfe) = dR_dρ(ρ, θ, R0fe, ϵfe, c0fe, cfe, sfe)
function dR_dρ(ρ::Real, θ::Real, R0fe::FE_rep, ϵfe::FE_rep, c0fe::FE_rep, cfe::AbstractVector{<:FE_rep}, sfe::AbstractVector{<:FE_rep})
    θr = Tr(ρ, θ, c0fe, cfe, sfe)
    dR_dρ = R0fe(ρ) + (R0fe(ρ) * D(ϵfe, ρ) + D(R0fe, ρ) * ϵfe(ρ)) * cos(θr)
    dR_dρ -= R0fe(ρ) * ϵfe(ρ) * sin(θr) * dTr_dρ(ρ, θ, c0fe, cfe, sfe)
    return dR_dρ
end

dZ_dρ(ρ, θ, R0fe, Z0fe, ϵfe, κfe, c0fe, cfe, sfe) = dZ_dρ(ρ, θ, R0fe, Z0fe, ϵfe, κfe)
function dZ_dρ(ρ::Real, θ::Real, R0fe::FE_rep, Z0fe::FE_rep, ϵfe::FE_rep, κfe::FE_rep)
    dZ_dρ = D(R0fe, ρ) * ϵfe(ρ) * κfe(ρ) + R0fe(ρ) * D(ϵfe, ρ) * κfe(ρ) + R0fe(ρ) *  ϵfe(ρ) * D(κfe, ρ)
    dZ_dρ = D(Z0fe, ρ) - dZ_dρ * sin(θ)
    return dZ_dρ
end

dR_dθ(ρ, θ, R0fe, Z0fe, ϵfe, κfe, c0fe, cfe, sfe) = dR_dθ(ρ, θ, R0fe, ϵfe, c0fe, cfe, sfe)
function dR_dθ(ρ::Real, θ::Real, R0fe::FE_rep, ϵfe::FE_rep, c0fe::FE_rep, cfe::AbstractVector{<:FE_rep}, sfe::AbstractVector{<:FE_rep})
    return -R0fe(ρ) * ϵfe(ρ) * sin(Tr(ρ, θ, c0fe, cfe, sfe)) * dTr_dθ(ρ, θ, cfe, sfe)
end

dZ_dθ(ρ, θ, R0fe, Z0fe, ϵfe, κfe, c0fe, cfe, sfe) = dZ_dθ(ρ, θ, R0fe, ϵfe, κfe)
function dZ_dθ(ρ::Real, θ::Real, R0fe::FE_rep, ϵfe::FE_rep, κfe::FE_rep)
    return -R0fe(ρ) * ϵfe(ρ) * κfe(ρ) * cos(θ)
end

function Jacobian(ρ::Real, θ::Real, R0fe::FE_rep, Z0fe::FE_rep, ϵfe::FE_rep, κfe::FE_rep, c0fe::FE_rep, cfe::AbstractVector{<:FE_rep}, sfe::AbstractVector{<:FE_rep})
    J = dR_dθ(ρ, θ, R0fe, ϵfe, c0fe, cfe, sfe) * dZ_dρ(ρ, θ, R0fe, Z0fe, ϵfe, κfe)
    J -= dZ_dθ(ρ, θ, R0fe, ϵfe, κfe) * dR_dρ(ρ, θ, R0fe, ϵfe, c0fe, cfe, sfe)
    return R_MXH(θ, MXH(R0fe, Z0fe, ϵfe, κfe, c0fe, cfe, sfe)) * J
end

function ∇ρ(ρ::Real, θ::Real, R0fe::FE_rep, Z0fe::FE_rep, ϵfe::FE_rep, κfe::FE_rep, c0fe::FE_rep, cfe::AbstractVector{<:FE_rep}, sfe::AbstractVector{<:FE_rep})
    gr2 = (R_MXH(θ, MXH(R0fe, Z0fe, ϵfe, κfe, c0fe, cfe, sfe)) / Jacobian(ρ, θ, R0fe, Z0fe, ϵfe, κfe, c0fe, cfe, sfe))^2
    gr2 *= dR_dθ(ρ, θ, R0fe, ϵfe, c0fe, cfe, sfe)^2 + dZ_dθ(ρ, θ, R0fe, ϵfe, κfe)^2
    return sqrt(gr2)
end

function ∇θ(ρ::Real, θ::Real, R0fe::FE_rep, Z0fe::FE_rep, ϵfe::FE_rep, κfe::FE_rep, c0fe::FE_rep, cfe::AbstractVector{<:FE_rep}, sfe::AbstractVector{<:FE_rep})
    gt2 = (R_MXH(θ, MXH(R0fe, Z0fe, ϵfe, κfe, c0fe, cfe, sfe)) / Jacobian(ρ, θ, R0fe, Z0fe, ϵfe, κfe, c0fe, cfe, sfe))^2
    gt2 *= dR_dρ(ρ, θ, R0fe, ϵfe, c0fe, cfe, sfe)^2 + dZ_dρ(ρ, θ, R0fe, Z0fe, ϵfe, κfe)^2
    return sqrt(gt2)
end