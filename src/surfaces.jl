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

    R0 = FE(ρ, rtype.(R0s))
    Z0 = FE(ρ, rtype.(Z0s))
    ϵ = FE(ρ, rtype.(ϵs))
    κ = FE(ρ, rtype.(κs))
    c0 = FE(ρ, rtype.(c0s))
    c = [FE(ρ, rtype.(cs[:, m])) for m in 1:M_mxh]
    s = [FE(ρ, rtype.(ss[:, m])) for m in 1:M_mxh]

    return R0, Z0, ϵ, κ, c0, c, s
end

function R_Z(R0::FE_rep, Z0::FE_rep, ϵ::FE_rep, κ::FE_rep, c0::FE_rep,
             c::AbstractVector{<:FE_rep}, s::AbstractVector{<:FE_rep}, x::Real, t::Real)
    cx = [cm(x) for cm in c]
    sx = [sm(x) for sm in s]
    surface = MXH(R0(x), Z0(x), ϵ(x), κ(x), c0(x), cx, sx)
    return surface(t)
end

R_Z(shot::Shot, x::Real, t::Real) = R_Z(surfaces_FE(shot)..., x, t)

function ρ_θ(R0::FE_rep, Z0::FE_rep, ϵ::FE_rep, κ::FE_rep, c0::FE_rep,
             c::AbstractVector{<:FE_rep}, s::AbstractVector{<:FE_rep}, R::Real, Z::Real)
    a = R0(1) * ϵ(1)
    function Δ!(F, x)
        ρ = 0.5*(tanh(x[1]) + 1.0)
        θ = π*(tanh(x[2]) + 1.0)
        X, Y = R_Z(R0, Z0, ϵ, κ, c0, c, s, ρ, θ)
        F[1] = X - R
        F[2] = Y - Z
    end
    # function j!(J, x)
    #     println("J: ", x)
    #     println((R0(x[1]), Z0(x[1]), ϵ(x[1]), κ(x[1]), c0(x[1])))
    #     J[1,1] = dR_dρ(x[1], x[2], R0, ϵ, c0, c, s)
    #     J[2,1] = dR_dθ(x[1], x[2], R0, ϵ, c0, c, s)
    #     J[1,2] = dZ_dρ(x[1], x[2], R0, Z0, ϵ, κ)
    #     J[2,2] = dZ_dθ(x[1], x[2], R0, ϵ, κ)
    #     println(J)
    #     println()
    # end
    x1_0 = min(sqrt(((R-R0(1))/a)^2 + ((Z-Z0(1))/(κ(1)*a))^2), 0.99)
    x1_0 = atanh(2 * x1_0 - 1.0)
    x2_0 = atan(Z-Z0(1), R-R0(1))
    x2_0 < 0 && (x2_0 += 2π)
    x2_0 = atanh(x2_0 / π - 1.0)
    S =  NLsolve.nlsolve(Δ!, [x1_0, x2_0])#, autodiff=:forward)#, method=:newton)
    if NLsolve.converged(S)
        return 0.5 * (tanh(S.zero[1]) + 1.0), π * (tanh(S.zero[2]) + 1.0)
    else
        return [NaN, NaN]
    end
end

ρ_θ(shot::Shot, R::Real, Z::Real) = ρ_θ(surfaces_FE(shot)..., R, Z)

Tr(ρ, θ, R0, Z0, ϵ, κ, c0, c, s) = Tr(ρ, θ, c0, c, s)
function Tr(ρ::Real, θ::Real, c0::FE_rep, c::AbstractVector{<:FE_rep}, s::AbstractVector{<:FE_rep})
    return θ + c0(ρ) + sum(dot((s[m](ρ), c[m](ρ)), sincos(m * θ)) for m in eachindex(c))
end

dTr_dρ(ρ, θ, R0, Z0, ϵ, κ, c0, c, s) = dTr_dρ(ρ, θ, c0, c, s)
function dTr_dρ(ρ::Real, θ::Real, c0::FE_rep, c::AbstractVector{<:FE_rep}, s::AbstractVector{<:FE_rep})
    return D(c0, ρ) + sum(dot((D(s[m], ρ), D(c[m], ρ)), sincos(m * θ)) for m in eachindex(c))
end

dTr_dθ(ρ, θ, R0, Z0, ϵ, κ, c0, c, s) = dTr_dθ(ρ, θ, c, s)
function dTr_dθ(ρ::Real, θ::Real, c::AbstractVector{<:FE_rep}, s::AbstractVector{<:FE_rep})
    return 1.0 + sum(dot((-c[m](ρ), s[m](ρ)), sincos(m * θ)) for m in eachindex(c))
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