function concentric_surface(x::Real, boundary::MXH)
    surface = deepcopy(boundary)
    # these go to zero as you go to axis
    surface.ϵ *= x
    surface.c *= x
    surface.s *= x
    return surface
end

function surfaces_FE(shot::Shot)

    R0s = zeros(shot.N)
    Z0s = zeros(shot.N)
    ϵs  = zeros(shot.N)
    κs  = zeros(shot.N)
    c0s = zeros(shot.N)
    cs  = zeros(shot.N, shot.M)
    ss  = zeros(shot.N, shot.M)

    for (i, surface) in enumerate(shot.surfaces)
        R0s[i] = surface.R0
        Z0s[i] = surface.Z0
        ϵs[i] = surface.ϵ
        κs[i] = surface.κ
        c0s[i] = surface.c0
        cs[i,:] = surface.c
        ss[i,:] = surface.s
    end

    rtype = typeof(shot.ρ[1])

    R0 = FE(shot.ρ, rtype.(R0s))
    Z0 = FE(shot.ρ, rtype.(Z0s))
    ϵ = FE(shot.ρ, rtype.(ϵs))
    κ = FE(shot.ρ, rtype.(κs))
    c0 = FE(shot.ρ, rtype.(c0s))
    c = [FE(shot.ρ, rtype.(cs[:, m])) for m in 1:shot.M]
    s = [FE(shot.ρ, rtype.(ss[:, m])) for m in 1:shot.M]

    return R0, Z0, ϵ, κ, c0, c, s
end

function R_Z(R0::FE_rep{T}, Z0::FE_rep{T}, ϵ::FE_rep{T}, κ::FE_rep{T}, c0::FE_rep{T},
             c::AbstractVector{FE_rep{T}}, s::AbstractVector{FE_rep{T}}, x::Real, t::Real) where T<:Real
    M = length(c)
    cx = [c[m](x) for m in 1:M] 
    sx = [s[m](x) for m in 1:M] 
    surface = MXH(R0(x), Z0(x), ϵ(x), κ(x), c0(x), cx, sx)
    return surface(t)
end

R_Z(shot::Shot, x::Real, t::Real) = R_Z(surfaces_FE(shot)..., x, t)

function ρ_θ(R0::FE_rep{T}, Z0::FE_rep{T}, ϵ::FE_rep{T}, κ::FE_rep{T}, c0::FE_rep{T},
             c::AbstractVector{FE_rep{T}}, s::AbstractVector{FE_rep{T}}, R::Real, Z::Real) where T<:Real
    a = R0(1) * ϵ(1)
    function Δ!(F, x)
        X, Y = R_Z(R0, Z0, ϵ, κ, c0, c, s, x[1], x[2]) 
        F[1] = X - R
        F[2] = Y - Z
    end
    S =  NLsolve.nlsolve(Δ!, [min(sqrt(((R-R0(1))/a)^2 + ((Z-Z0(1))/(κ(1)*a))^2), 0.99), atan(Z-Z0(1), R-R0(1))])
    if NLsolve.converged(S)
        return S.zero
    else
        return [NaN, NaN]
    end
end

ρ_θ(shot::Shot, R::Real, Z::Real) = ρ_θ(surfaces_FE(shot)..., R, Z)