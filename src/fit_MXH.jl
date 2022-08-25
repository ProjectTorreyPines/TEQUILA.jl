Ψres(Ψ, R::Real, Z::Real, level::Real) = (Ψ(R, Z) - level)^2

function res_Rext(x::AbstractVector{<:Real}, Ψ, level::Real, R0::Real; weight::Real=1e-6)
    res_Ψ = Ψres(Ψ, x[1], x[2], level)
    res_R = (x[1] - R0)^2
    res = res_Ψ - weight * res_R
    return res
end

function res_Zext(x::AbstractVector{<:Real}, Ψ, level::Real, Z0::Real; weight::Real=1e-6)
    res_Ψ = Ψres(Ψ, x[1], x[2], level)
    res_Z = (x[2] - Z0)^2
    res = res_Ψ - weight * res_Z
    return res
end

function find_extrema(Ψ, level::Real, R0::Real, Z0::Real, a::Real, b::Real; weight::Real=1e-6, method=Optim.ConjugateGradient(linesearch = LineSearches.BackTracking()))

    fr(x) = res_Rext(x, Ψ, level, R0; weight)
    fz(x) = res_Zext(x, Ψ, level, Z0; weight)

    lower_2[1] = R0
    upper_2[1] = R0 + a
    x0_2[1] = 0.5*(lower_2[1] + upper_2[1])
    lower_2[2] = Z0 - b
    upper_2[2] = Z0 + b
    x0_2[2] = Z0
    res = Optim.optimize(fr, lower_2, upper_2, x0_2, Optim.Fminbox(method))#; autodiff=:forward)
    (Rmax, Z_at_Rmax) = res.minimizer

    lower_2[1] = R0 - a
    upper_2[1] = R0
    x0_2[1] = 0.5*(lower_2[1] + upper_2[1])
    #lower_2[2] = Z0 - b
    #upper_2[2] = Z0 + b
    #x0_2[2] = Z0
    res = Optim.optimize(fr, lower_2, upper_2, x0_2, Optim.Fminbox(method))#; autodiff=:forward)
    (Rmin, Z_at_Rmin) = res.minimizer

    lower_2[1] = Rmin
    upper_2[1] = Rmax
    x0_2[1] = 0.5*(lower_2[1] + upper_2[1])
    lower_2[2] = max(Z_at_Rmin, Z_at_Rmax)
    #upper_2[2] = Z0 + b
    x0_2[2] = 0.5*(lower_2[2] + upper_2[2])
    res = Optim.optimize(fz, lower_2, upper_2, x0_2, Optim.Fminbox(method))#; autodiff=:forward)
    (R_at_Zmax, Zmax) = res.minimizer

    #lower_2[1] = Rmin
    #upper_2[1] = Rmax
    #x0_2[1] = 0.5*(lower_2[1] + upper_2[1])
    lower_2[2] = Z0 - b
    upper_2[2] = min(Z_at_Rmin, Z_at_Rmax)
    x0_2[2] = 0.5*(lower_2[2] + upper_2[2])
    res = Optim.optimize(fz, lower_2, upper_2, x0_2, Optim.Fminbox(method))#; autodiff=:forward)
    (R_at_Zmin, Zmin) = res.minimizer

    return (R_at_Zmax, Zmax), (R_at_Zmin, Zmin), (Rmax, Z_at_Rmax), (Rmin, Z_at_Rmin)
end

function Δθr_grid!(Δθr::AbstractVector{<:Number}, Ψ, level::Real, M::Integer, θz::AbstractVector{<:Real},
                  R0::Real, Z0::Real, a::Real, b::Real,
                  Rmax::Real, Z_at_Rmax::Real, Rmin::Real, Z_at_Rmin::Real, R_at_Zmax::Real, R_at_Zmin::Real)

    # Bottom point
    k = length(θz)
    Δθr[k] = acos((R_at_Zmin - R0) / a) - halfpi

    # Top point
    kk = 2M + 6 - k
    Δθr[kk] = -acos((R_at_Zmax - R0) / a) + halfpi

    @views for (k, θ) in enumerate(θz[1:end-1])

        bb = b * sin(θ)

        # LOWER
        Z = Z0 - bb
        flow(R) = Ψres(Ψ, R, Z, level)

        # outboard
        Ro = Optim.optimize(flow, R_at_Zmin, Rmax).minimizer
        Δθr[k] = sign(Z_at_Rmax - Z) * acos((Ro - R0) / a) - θ

        # inboard
        kk = M + 4 - k
        Ri = Optim.optimize(flow, Rmin, R_at_Zmin).minimizer
        Δθr[kk] = sign(Z_at_Rmin - Z) * acos((Ri - R0) / a) - (π - θ)

        if θ != 0
            # UPPER
            Z = Z0 + bb
            fup(R) = Ψres(Ψ, R, Z, level)

            # outboard
            kk = 2M + 6 - k
            Ro = Optim.optimize(fup, R_at_Zmax, Rmax).minimizer
            Δθr[kk] = sign(Z_at_Rmax - Z) * acos((Ro - R0) / a) + θ

            # inboard
            kk = M + 2 + k
            Ri = Optim.optimize(fup, Rmin, R_at_Zmax).minimizer
            Δθr[kk] = sign(Z_at_Rmin - Z) * acos((Ri - R0) / a) + (π - θ)

        end
    end
    @views Δθr[real.(Δθr) .< -π] .+= twopi
    @views Δθr[real.(Δθr) .>= π] .-= twopi
    return Δθr
end

"""
Fit one surface
"""
function MXH_surface(Ψ, level::Real, M::Integer, mxh_init::MXH)

    R0_guess = mxh_init.R0
    Z0_guess = mxh_init.Z0
    a_guess  = R0_guess * mxh_init.ϵ
    b_guess  = a_guess * mxh_init.κ

    (R_at_Zmax, Zmax), (R_at_Zmin, Zmin), (Rmax, Z_at_Rmax), (Rmin, Z_at_Rmin) = find_extrema(Ψ, level, R0_guess, Z0_guess, a_guess, b_guess)

    R0 = 0.5 * (Rmax + Rmin)
    Z0 = 0.5 * (Zmax + Zmin)
    a  = Rmax - R0
    ϵ  = a / R0
    b  = Zmax - Z0
    κ  = b / a

    # Roughly twice as many points as needed to reduce error at high m
    MM = 2M
    θz = range(0, halfpi, MM ÷ 2 + 2)
    Δθr = zeros(ComplexF64, 2MM + 4)

    Δθr_grid!(Δθr, Ψ, level, MM, θz, R0, Z0, a, b, Rmax, Z_at_Rmax, Rmin, Z_at_Rmin, R_at_Zmax, R_at_Zmin)
    fft!(Δθr)
    invMM2 = 1.0 / (MM + 2)
    c0 = 0.5 * real(Δθr[1]) * invMM2
    @views c =  real.(Δθr[2:(M + 1)]) .* invMM2
    @views s = -imag.(Δθr[2:(M + 1)]) .* invMM2  # fft sign convention

    @views mxh = MXH(R0, Z0, ϵ, κ, c0, c, s)
    return mxh
end

"""
Fit all surfaces for each ρ (using ψnorm=ρ^2 approximation)
"""
function loop_surfaces!(Ψ, levels::AbstractVector{<:Real}, M::Integer, surfaces::Vector{<:MXH}, bnd::MXH)
    N = length(surfaces)
    @views for k in reverse(eachindex(surfaces))
        k == 1 && continue
        surfaces[k] = k==N ? deepcopy(bnd) : MXH_surface(Ψ, levels[k], M,  surfaces[k+1])
    end
end

function refit(shot, levels::AbstractVector{<:Real}, bnd::MXH)

    N = length(levels)

    Ψ(r, z) = shot(r, z)

    surfaces = Vector{typeof(bnd)}(undef, N)

    # Allocate initial guess and bounds
    M = length(bnd.c)
    loop_surfaces!(Ψ, levels, M, surfaces, bnd)

    # Extrapolate or set to zero on-axis
    ρ2 = sqrt((levels[2]-levels[1])/(levels[end]-levels[1]))
    ρ3 = sqrt((levels[3]-levels[1])/(levels[end]-levels[1]))
    h = 1.0 / (ρ3 - ρ2)
    R0 = h .* (surfaces[2].R0 .* ρ3 .- surfaces[3].R0 .* ρ2)
    Z0 = h .* (surfaces[2].Z0 .* ρ3 .- surfaces[3].Z0 .* ρ2)
    κ  = h .* (surfaces[2].κ  .* ρ3 .- surfaces[3].κ  .* ρ2)
    c0 = h .* (surfaces[2].c0 .* ρ3 .- surfaces[3].c0 .* ρ2)
    surfaces[1] = MXH(R0, Z0, 0.0, κ, c0, zeros(M), zeros(M))

    shot_refit = Shot(shot.N, shot.M, shot.ρ, surfaces, Ψ)
    return shot_refit
end