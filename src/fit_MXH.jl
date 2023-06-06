function find_extrema(shot, level::Real, Ψ0::Real)

    model = Model(NLopt.Optimizer)
    set_optimizer_attribute(model, "algorithm", :LD_SLSQP)
    @variable(model, ρ)
    @variable(model, θ)

    psi = (r, t) -> psi_ρθ(shot, r, t)
    register(model, :psi, 2, (r, t) -> psi_ρθ(shot, r, t), autodiff=true)
    @NLconstraint(model, psi(ρ, θ) == level)

    set_start_value(ρ, sqrt(1 - level / Ψ0))

    set_lower_bound(ρ, 0.0)
    set_upper_bound(ρ, 1.0)
    set_lower_bound(θ, -π)
    set_upper_bound(θ, π)

    Rloc = (r, t) -> TEQUILA.R(shot, r, t)
    Zloc = (r, t) -> TEQUILA.Z(shot, r, t)
    register(model, :Rloc, 2, Rloc, autodiff=true)
    register(model, :Zloc, 2, Zloc, autodiff=true)

    # Zmax
    set_start_value(θ, -π/2)
    @NLobjective(model, Max, Zloc(ρ, θ))
    JuMP.optimize!(model)
    ρ_Zmax = value(ρ)
    θ_Zmax = value(θ)

    # Zmin
    set_start_value(θ, π/2)
    @NLobjective(model, Min, Zloc(ρ, θ))
    JuMP.optimize!(model)
    ρ_Zmin = value(ρ)
    θ_Zmin = value(θ)

    # Rmax
    set_start_value(θ, 0.0)
    @NLobjective(model, Max, Rloc(ρ, θ))
    JuMP.optimize!(model)
    ρ_Rmax = value(ρ)
    θ_Rmax = value(θ)

    # Rmin
    set_lower_bound(θ, 0)
    set_upper_bound(θ, 2π)
    set_start_value(θ, π)
    @NLobjective(model, Min, Rloc(ρ, θ))
    JuMP.optimize!(model)
    ρ_Rmin = value(ρ)
    θ_Rmin = value(θ)

    return R_Z(shot, ρ_Rmax, θ_Rmax)..., R_Z(shot, ρ_Rmin, θ_Rmin)..., R_Z(shot, ρ_Zmax, θ_Zmax)..., R_Z(shot, ρ_Zmin, θ_Zmin)...
end

function fit_MXH!(flat, shot, lvl, Ψ0, Fi, Fo, P)

    Rmax, Z_at_Rmax, Rmin, Z_at_Rmin, R_at_Zmax, Zmax, R_at_Zmin, Zmin  = find_extrema(shot, lvl , Ψ0)

    R0 = 0.5 * (Rmax + Rmin)
    a  = 0.5 * (Rmax - Rmin)
    ϵ  = a / R0
    Z0 = 0.5 * (Zmax + Zmin)
    b  = 0.5 * (Zmax - Zmin)
    κ  = b / a

    M = length(shot.cfe)

    invM2 = 1.0 / (M + 2)
    Δθ = π * invM2
    θs = 0.0:Δθ:(2π-Δθ)

    branch = Z_at_Rmax > Z0 ? 1 : 0

    for (k, θ) in enumerate(θs)
        z = Z0 - b * sin(θ)

        if θ < halfpi
            # lower right
            r = Roots.find_zero(x -> shot(x, z) - lvl, (R_at_Zmin, Rmax), Roots.A42())
            (branch == 0 && z < Z_at_Rmax) && (branch = 1)
        elseif θ == halfpi
            r = R_at_Zmin
            branch = 2
        elseif (θ > halfpi) && (θ <= π)
            # lower left
            r = Roots.find_zero(x -> shot(x, z) - lvl, (Rmin, R_at_Zmin), Roots.A42())
            branch = (z <= Z_at_Rmin) ? 2 : 3
            #(branch == 2 && z > Z_at_Rmin) && (branch = 3)
        elseif θ < 3 * halfpi
            # upper left
            r = Roots.find_zero(x -> shot(x, z) - lvl, (Rmin, R_at_Zmax), Roots.A42())
            branch = (z <= Z_at_Rmin) ? 2 : 3
        elseif θ == 3 * halfpi
            r = R_at_Zmax
            branch = 4
        elseif (θ > 3 * halfpi) && (θ <= 2π)
            # upper right
            r = Roots.find_zero(x -> shot(x, z) - lvl, (R_at_Zmax, Rmax), Roots.A42())
            branch = (z > Z_at_Rmax) ? 4 : 5
        end

        θᵣ = acos((r - R0) / a)
        if branch == 0
            θᵣ = -θᵣ
        elseif branch == 1 || branch == 2
            θᵣ = θᵣ
        elseif branch == 3 || branch == 4
            θᵣ = 2π - θᵣ
        elseif branch == 5
            θᵣ = 2π + θᵣ
        end
        Fi[k] = θᵣ - θ
    end

    flat[1] = R0
    flat[2] = Z0
    flat[3] = ϵ
    flat[4] = κ

    mul!(Fo, P, Fi)
    flat[5] = 0.5 * real(Fo[1]) * invM2
    @views flat[6:(5+M)] .= real.(Fo[2:(M+1)]) .* invM2
    @views flat[(6+M):(5+2M)] .= -imag.(Fo[2:(M+1)]) .* invM2 # fft sign convention
    return flat

end

function refit(shot::Shot, lvls::AbstractVector{<:Real}, Ψaxis::Real)

    L = length(shot.cfe)

    Fis, _, Fos, Ps = fft_prealloc_threaded(L)

    surfaces = deepcopy(shot.surfaces)

    @Threads.threads for (k, lvl) in collect(enumerate(lvls))
        (k == 1 || k == length(lvls)) && continue
        tid = Threads.threadid()
        Fi = Fis[tid]
        Fo = Fos[tid]
        P  = Ps[tid]

        @views flat = surfaces[:, k]

        fit_MXH!(flat, shot, lvl, Ψaxis, Fi, Fo, P)
    end

    # Extrapolate or set to zero on-axis
    ρ2 = sqrt((lvls[2]-lvls[1])/(lvls[end]-lvls[1]))
    ρ3 = sqrt((lvls[3]-lvls[1])/(lvls[end]-lvls[1]))
    h = 1.0 / (ρ3 - ρ2)
    surfaces[1, 1] = h .* (surfaces[1, 2] .* ρ3 .- surfaces[1, 3] .* ρ2)
    surfaces[2, 1] = h .* (surfaces[2, 2] .* ρ3 .- surfaces[2, 3] .* ρ2)
    surfaces[3, 1] = 0.0
    surfaces[4, 1] = h .* (surfaces[4, 2] .* ρ3 .- surfaces[4, 3] .* ρ2)
    surfaces[5, 1] = h .* (surfaces[5, 2] .* ρ3 .- surfaces[5, 3] .* ρ2)
    @views surfaces[6:end, 1] .= 0.0

    #remap_shot!(shot, surfaces)
    shot_refit = Shot(shot.N, shot.M, shot.ρ, surfaces, shot;
                      P = shot.P, dP_dψ = shot.dP_dψ,
                      F_dF_dψ = shot.F_dF_dψ, Jt_R = shot.Jt_R, Jt = shot.Jt,
                      Pbnd = shot.Pbnd, Fbnd = shot.Fbnd, Ip_target = shot.Ip_target)

    return shot_refit

end