function concentric_surface(x::Real, boundary::MXH)
    surface = deepcopy(boundary)
    # these go to zero as you go to axis
    surface.ϵ *= x
    if x < 1.0
        surface.c *= x
        surface.s *= x
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
    Cs  = zeros(N, M_mxh)
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

    R0 = FE(ρ, rtype.(R0s))
    Z0 = FE(ρ, rtype.(Z0s))
    ϵ = FE(ρ, rtype.(ϵs))
    κ = FE(ρ, rtype.(κs))
    c0 = FE(ρ, rtype.(c0s))
    c = [FE(ρ, rtype.(cs[:, m])) for m in 1:M_mxh]
    s = [FE(ρ, rtype.(ss[:, m])) for m in 1:M_mxh]

    return R0, Z0, ϵ, κ, c0, c, s
end

function surfaces_FE(ρ:: AbstractVector{<:Real}, surfaces:: AbstractMatrix{<:Real} )

    rtype = typeof(ρ[1])

    @views R0 = FE(ρ, rtype.(surfaces[1, :]))
    @views Z0 = FE(ρ, rtype.(surfaces[2, :]))
    @views ϵ  = FE(ρ, rtype.(surfaces[3, :]))
    @views κ  = FE(ρ, rtype.(surfaces[4, :]))
    @views c0 = FE(ρ, rtype.(surfaces[5, :]))

    M = (size(surfaces,1) - 5) ÷ 2
    @views c  = [FE(ρ, rtype.(surfaces[5 + m, :])) for m in 1:M]
    @views s  = [FE(ρ, rtype.(surfaces[5 + m + M, :])) for m in 1:M]

    return R0, Z0, ϵ, κ, c0, c, s
end

R_Z(shot::Shot, x::Real, t::Real) = R_Z(surfaces_FE(shot)..., x, t)

function R_Z(R0::FE_rep, Z0::FE_rep, ϵ::FE_rep, κ::FE_rep, c0::FE_rep,
             c::AbstractVector{<:FE_rep}, s::AbstractVector{<:FE_rep}, x::Real, t::Real)
    R0x = R0(x)
    Z0x = Z0(x)
    ϵx  = ϵ(x)
    κx  = κ(x)
    c0x = c0(x)
    cx = [cm(x) for cm in c]
    sx = [sm(x) for sm in s]
    a = R0x * ϵx

    R = R_MXH(t, R0x, ϵx, c0x, cx, sx, a)
    Z = Z_MXH(t, R0x, Z0x, ϵx, κx, a)
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
    θo = asin(aa)
    signθ = θo < 0.0 ? -1.0 : 1.0
    θi = signθ * π - θo

    c0 = evaluate_inbounds(c0fe, k, nu_ou, nu_eu, nu_ol, nu_el)

    θro = θo + c0
    θri = θi + c0
    @inbounds for m in eachindex(cfe)
        S = evaluate_inbounds(sfe[m], k, nu_ou, nu_eu, nu_ol, nu_el)
        C = evaluate_inbounds(cfe[m], k, nu_ou, nu_eu, nu_ol, nu_el)
        θro += dot((S, C), sincos(m * θo))
        θri += dot((S, C), sincos(m * θi))
    end
    dR = R0 - R
    reso =  (dR + a * cos(θro))^2
    resi =  (dR + a * cos(θri))^2
    if reso < resi
        return_θ ? (return θo) : (return reso)
    else
        return_θ ? (return θi) : (return resi)
    end
end

function res_zext(ρ::Real, R0fe::FE_rep, Z0fe::FE_rep, ϵfe::FE_rep, κfe::FE_rep, Z::Real)
    k, nu_ou, nu_eu, nu_ol, nu_el = compute_bases(R0fe.x, ρ)
    R0 = evaluate_inbounds(R0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    Z0 = evaluate_inbounds(Z0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ϵ = evaluate_inbounds(ϵfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    κ = evaluate_inbounds(κfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    b = R0 * ϵ * κ
    return (Z0 + sign(Z-Z0) * b - Z)^2
end


function ρθ_RZ(shot::Shot, R::Real, Z::Real, R0::FE_rep, Z0::FE_rep, ϵ::FE_rep, κ::FE_rep, c0::FE_rep,
               c::AbstractVector{<:FE_rep}, s::AbstractVector{<:FE_rep})

    ki, ρi, θi, ko, ρo, θo = surface_bracket(shot, R, Z)
    ki==ko && return ρo, θo # on a surface exactly

    if abs(θi) == 0.5 * π
        # find ρ where Z = Zext
        f_zext(x) = res_zext(x, R0, Z0, ϵ, κ, Z)
        ρi = optimize(f_zext, ρi, ρo).minimizer
    end

    f_find_ρ(x) = res_find_ρ(x, R0, Z0, ϵ, κ, c0, c, s, R, Z)

    ρ = optimize(f_find_ρ, ρi, ρo).minimizer

    θ = res_find_ρ(ρ, R0, Z0, ϵ, κ, c0, c, s, R, Z, return_θ=true)

    return ρ, θ
end

Tr(ρ, θ, R0, Z0, ϵ, κ, c0, c, s) = Tr(ρ, θ, c0, c, s)
function Tr(ρ::Real, θ::Real, c0::FE_rep, c::AbstractVector{<:FE_rep}, s::AbstractVector{<:FE_rep})
    θr = θ + c0(ρ)
    @inbounds for m in eachindex(c)
        S = s[m](ρ)
        C = c[m](ρ)
        θr += dot((S, C), sincos(m * θ))
    end
    return θr
end

dTr_dρ(ρ, θ, R0, Z0, ϵ, κ, c0, c, s) = dTr_dρ(ρ, θ, c0, c, s)
function dTr_dρ(ρ::Real, θ::Real, c0::FE_rep, c::AbstractVector{<:FE_rep}, s::AbstractVector{<:FE_rep})
    dθr_dρ = D(c0, ρ)
    @inbounds for m in eachindex(c)
        dS = D(s[m], ρ)
        dC = D(c[m], ρ)
        dθr_dρ += dot((dS, dC), sincos(m * θ))
    end
    return dθr_dρ
end

dTr_dθ(ρ, θ, R0, Z0, ϵ, κ, c0, c, s) = dTr_dθ(ρ, θ, c, s)
function dTr_dθ(ρ::Real, θ::Real, c::AbstractVector{<:FE_rep}, s::AbstractVector{<:FE_rep})
    dθr_dθ = 1.0
    @inbounds for m in eachindex(c)
        S = s[m](ρ)
        C = c[m](ρ)
        dθr_dθ += dot((-C, S), sincos(m * θ))
    end
    return dθr_dθ
end

dR_dρ(ρ, θ, R0, Z0, ϵ, κ, c0, c, s) = dR_dρ(ρ, θ, R0, ϵ, c0, c, s)
function dR_dρ(ρ::Real, θ::Real, R0::FE_rep, ϵ::FE_rep, c0::FE_rep, c::AbstractVector{<:FE_rep}, s::AbstractVector{<:FE_rep})
    θr = Tr(ρ, θ, c0, c, s)
    dR_dρ = R0(ρ) + (R0(ρ) * D(ϵ, ρ) + D(R0, ρ) * ϵ(ρ)) * cos(θr)
    dR_dρ -= R0(ρ) * ϵ(ρ) * sin(θr) * dTr_dρ(ρ, θ, c0, c, s)
    return dR_dρ
end

dZ_dρ(ρ, θ, R0, Z0, ϵ, κ, c0, c, s) = dZ_dρ(ρ, θ, R0, Z0, ϵ, κ)
function dZ_dρ(ρ::Real, θ::Real, R0::FE_rep, Z0::FE_rep, ϵ::FE_rep, κ::FE_rep)
    dZ_dρ = D(R0, ρ) * ϵ(ρ) * κ(ρ) + R0(ρ) * D(ϵ, ρ) * κ(ρ) + R0(ρ) *  ϵ(ρ) * D(κ, ρ)
    dZ_dρ = D(Z0, ρ) - dZ_dρ * sin(θ)
    return dZ_dρ
end

dR_dθ(ρ, θ, R0, Z0, ϵ, κ, c0, c, s) = dR_dθ(ρ, θ, R0, ϵ, c0, c, s)
function dR_dθ(ρ::Real, θ::Real, R0::FE_rep, ϵ::FE_rep, c0::FE_rep, c::AbstractVector{<:FE_rep}, s::AbstractVector{<:FE_rep})
    return -R0(ρ) * ϵ(ρ) * sin(Tr(ρ, θ, c0, c, s)) * dTr_dθ(ρ, θ, c, s)
end

dZ_dθ(ρ, θ, R0, Z0, ϵ, κ, c0, c, s) = dZ_dθ(ρ, θ, R0, ϵ, κ)
function dZ_dθ(ρ::Real, θ::Real, R0::FE_rep, ϵ::FE_rep, κ::FE_rep)
    return -R0(ρ) * ϵ(ρ) * κ(ρ) * cos(θ)
end

function Jacobian(ρ::Real, θ::Real, R0::FE_rep, Z0::FE_rep, ϵ::FE_rep, κ::FE_rep, c0::FE_rep, c::AbstractVector{<:FE_rep}, s::AbstractVector{<:FE_rep})
    J = dR_dθ(ρ, θ, R0, ϵ, c0, c, s) * dZ_dρ(ρ, θ, R0, Z0, ϵ, κ)
    J -= dZ_dθ(ρ, θ, R0, ϵ, κ) * dR_dρ(ρ, θ, R0, ϵ, c0, c, s)
    return R_MXH(θ, MXH(R0, Z0, ϵ, κ, c0, c, s)) * J
end

function ∇ρ(ρ::Real, θ::Real, R0::FE_rep, Z0::FE_rep, ϵ::FE_rep, κ::FE_rep, c0::FE_rep, c::AbstractVector{<:FE_rep}, s::AbstractVector{<:FE_rep})
    gr2 = (R_MXH(θ, MXH(R0, Z0, ϵ, κ, c0, c, s)) / Jacobian(ρ, θ, R0, Z0, ϵ, κ, c0, c, s))^2
    gr2 *= dR_dθ(ρ, θ, R0, ϵ, c0, c, s)^2 + dZ_dθ(ρ, θ, R0, ϵ, κ)^2
    return sqrt(gr2)
end

function ∇θ(ρ::Real, θ::Real, R0::FE_rep, Z0::FE_rep, ϵ::FE_rep, κ::FE_rep, c0::FE_rep, c::AbstractVector{<:FE_rep}, s::AbstractVector{<:FE_rep})
    gt2 = (R_MXH(θ, MXH(R0, Z0, ϵ, κ, c0, c, s)) / Jacobian(ρ, θ, R0, Z0, ϵ, κ, c0, c, s))^2
    gt2 *= dR_dρ(ρ, θ, R0, ϵ, c0, c, s)^2 + dZ_dρ(ρ, θ, R0, Z0, ϵ, κ)^2
    return sqrt(gt2)
end