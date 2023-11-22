function valid_extrema(Rmax, Z_at_Rmax, Rmin, Z_at_Rmin, R_at_Zmax, Zmax, R_at_Zmin, Zmin)
    (R_at_Zmax <= Rmin || R_at_Zmax >= Rmax) && return false
    (R_at_Zmin <= Rmin || R_at_Zmin >= Rmax) && return false
    (Z_at_Rmax <= Zmin || Z_at_Rmax >= Zmax) && return false
    (Z_at_Rmin <= Zmin || Z_at_Rmin >= Zmax) && return false
    return true
end

function residual_extrema(shot, lvl, Rmax, Z_at_Rmax, Rmin, Z_at_Rmin, R_at_Zmax, Zmax, R_at_Zmin, Zmin)
    res =  (shot(Rmax, Z_at_Rmax) - lvl) ^ 2 + (shot(Rmin, Z_at_Rmin) - lvl) ^ 2
    res += (shot(R_at_Zmax, Zmax) - lvl) ^ 2 + (shot(R_at_Zmin, Zmin) - lvl) ^ 2
    return sqrt(res / lvl^2)
end

function find_extrema(shot, level::Real, Ψaxis::Real, Raxis::Real, Zaxis::Real, ρaxis::Real; algorithm::Symbol=:LD_SLSQP)

    model = Model(NLopt.Optimizer; add_bridges=false)
    set_optimizer_attribute(model, "algorithm", algorithm)
    set_optimizer_attribute(model, "maxtime", 10.0)
    @variable(model, ρ)
    @variable(model, θ)

    register(model, :psi, 2, (r, t) -> psi_ρθ(shot, r, t), autodiff=true)
    @NLconstraint(model, psi(ρ, θ) == level)

    ρguess = sqrt(1 - level / Ψaxis)
    aguess = ρguess * shot.surfaces[1, end] * shot.surfaces[2, end]
    bguess = aguess * shot.surfaces[4, end]

    set_lower_bound(ρ, 1e-8) #avoid singularity on-axis
    set_upper_bound(ρ, 1.0)

    Rloc = (r, t) -> TEQUILA.R(shot, r, t)
    Zloc = (r, t) -> TEQUILA.Z(shot, r, t)
    register(model, :Rloc, 2, Rloc, autodiff=true)
    register(model, :Zloc, 2, Zloc, autodiff=true)

    # Zmax
    if ρguess < 2ρaxis
        rguess = Raxis
        zguess = Zaxis + bguess
        ρ0, θ0 = ρθ_RZ(shot, rguess, zguess)
        set_start_value(ρ, ρ0)
        set_lower_bound(θ, θ0 - twopi)
        set_upper_bound(θ, θ0 + twopi)
        set_start_value(θ, θ0)
    else
        set_start_value(ρ, ρguess)
        set_lower_bound(θ, -2.5 * π)
        set_upper_bound(θ,  1.5 * π)
        set_start_value(θ, -0.5 * π)
    end
    @NLobjective(model, Max, Zloc(ρ, θ))
    JuMP.optimize!(model)
    @assert termination_status(model) in jump_success
    ρ_Zmax = value(ρ)
    θ_Zmax = value(θ)

    # Zmin
    if ρguess < 2ρaxis
        rguess = Raxis
        zguess = Zaxis - bguess
        ρ0, θ0 = ρθ_RZ(shot, rguess, zguess)
        set_start_value(ρ, ρ0)
        set_lower_bound(θ, θ0 - twopi)
        set_upper_bound(θ, θ0 + twopi)
        set_start_value(θ, θ0)
    else
        set_start_value(ρ, ρguess)
        set_lower_bound(θ, -1.5 * π)
        set_upper_bound(θ,  2.5 * π)
        set_start_value(θ,  0.5 * π)
    end
    @NLobjective(model, Min, Zloc(ρ, θ))
    JuMP.optimize!(model)
    @assert termination_status(model) in jump_success
    ρ_Zmin = value(ρ)
    θ_Zmin = value(θ)

    # Rmax
    if ρguess < 2ρaxis
        rguess = Raxis + aguess
        zguess = Zaxis
        ρ0, θ0 = ρθ_RZ(shot, rguess, zguess)
        set_start_value(ρ, ρ0)
        set_lower_bound(θ, θ0 - twopi)
        set_upper_bound(θ, θ0 + twopi)
        set_start_value(θ, θ0)
    else
        set_start_value(ρ, ρguess)
        set_lower_bound(θ, -twopi)
        set_upper_bound(θ, twopi)
        set_start_value(θ, 0.0)
    end
    @NLobjective(model, Max, Rloc(ρ, θ))
    JuMP.optimize!(model)
    @assert termination_status(model) in jump_success
    ρ_Rmax = value(ρ)
    θ_Rmax = value(θ)

    # Rmin
    if ρguess < 2ρaxis
        rguess = Raxis - aguess
        zguess = Zaxis
        ρ0, θ0 = ρθ_RZ(shot, rguess, zguess)
        set_start_value(ρ, ρ0)
        set_lower_bound(θ, θ0 - twopi)
        set_upper_bound(θ, θ0 + twopi)
        set_start_value(θ, θ0)
    else
        set_start_value(ρ, ρguess)
        set_lower_bound(θ, -π)
        set_upper_bound(θ, 3π)
        set_start_value(θ, π)
    end
    @NLobjective(model, Min, Rloc(ρ, θ))
    JuMP.optimize!(model)
    @assert termination_status(model) in jump_success
    ρ_Rmin = value(ρ)
    θ_Rmin = value(θ)

    Rmax, Z_at_Rmax = R_Z(shot, ρ_Rmax, θ_Rmax)
    Rmin, Z_at_Rmin = R_Z(shot, ρ_Rmin, θ_Rmin)
    R_at_Zmax, Zmax = R_Z(shot, ρ_Zmax, θ_Zmax)
    R_at_Zmin, Zmin = R_Z(shot, ρ_Zmin, θ_Zmin)

    return Rmax, Z_at_Rmax, Rmin, Z_at_Rmin, R_at_Zmax, Zmax, R_at_Zmin, Zmin
end

function find_extrema_RZ(shot, level::Real, Raxis::Real, Zaxis::Real)

    R0 = shot.surfaces[1, end]
    Z0 = shot.surfaces[2, end]
    a =  R0 * shot.surfaces[3, end]
    b = shot.surfaces[4, end] * a

    Rb_min = R0 - a
    Rb_max = R0 + a
    Zb_min = Z0 - b
    Zb_max = Z0 + b

    model = Model(NLopt.Optimizer; add_bridges=false)
    set_optimizer_attribute(model, "algorithm", :LN_COBYLA)
    set_optimizer_attribute(model, "maxtime", 10.0)
    @variable(model, R)
    @variable(model, Z)

    register(model, :psi, 2, (x, y) -> shot(x, y), autodiff=true)
    @NLconstraint(model, psi(R, Z) == level)

    # Zmax
    set_lower_bound(Z, Zaxis)
    set_upper_bound(Z, Zb_max)
    set_lower_bound(R, Rb_min)
    set_upper_bound(R, Rb_max)
    set_start_value(R, Raxis)
    set_start_value(Z, Zaxis)
    @objective(model, Max, Z)
    JuMP.optimize!(model)
    @assert termination_status(model) in jump_success
    R_at_Zmax = value(R)
    Zmax = value(Z)

    # Zmin
    set_lower_bound(Z, Zb_min)
    set_upper_bound(Z, Zaxis)
    set_lower_bound(R, Rb_min)
    set_upper_bound(R, Rb_max)
    set_start_value(R, Raxis)
    set_start_value(Z, Zaxis)
    @objective(model, Min, Z)
    JuMP.optimize!(model)
    @assert termination_status(model) in jump_success
    R_at_Zmin = value(R)
    Zmin = value(Z)

    # Rmax
    set_lower_bound(Z, Zb_min)
    set_upper_bound(Z, Zb_max)
    set_lower_bound(R, Raxis)
    set_upper_bound(R, Rb_max)
    set_start_value(R, Raxis)
    set_start_value(Z, Zaxis)
    @objective(model, Max, R)
    JuMP.optimize!(model)
    @assert termination_status(model) in jump_success
    Rmax = value(R)
    Z_at_Rmax = value(Z)

    # Rmin
    set_lower_bound(Z, Zb_min)
    set_upper_bound(Z, Zb_max)
    set_lower_bound(R, Rb_min)
    set_upper_bound(R, Raxis)
    set_start_value(R, Raxis)
    set_start_value(Z, Zaxis)
    @objective(model, Min, R)
    JuMP.optimize!(model)
    @assert termination_status(model) in jump_success
    Rmin = value(R)
    Z_at_Rmin = value(Z)

    return Rmax, Z_at_Rmax, Rmin, Z_at_Rmin, R_at_Zmax, Zmax, R_at_Zmin, Zmin
end

function find_r(shot, z, lvl, rmin, rmax)
    r = 0.0
    try
        r = Roots.find_zero(x -> shot(x, z) - lvl, (rmin, rmax), Roots.A42())
    catch err
        if isa(err, ArgumentError)
            if abs((shot(rmin, z) - lvl) / lvl) < 1e-4
                r = rmin
            elseif abs((shot(rmax, z) - lvl) / lvl) < 1e-4
                r = rmax
            else
                rethrow(err)
            end
        else
            rethrow(err)
        end
    end
    return r
end

function compute_thetar!(Fi, θs, shot, lvl, R0, Z0, a, b, Rmax, Z_at_Rmax, Rmin, Z_at_Rmin, R_at_Zmax, R_at_Zmin)

    branch = Z_at_Rmax > Z0 ? 1 : 0
    for (k, θ) in enumerate(θs)
        z = Z0 - b * sin(θ)
        r = 0.0
        if θ < halfpi
            # lower right
            r = find_r(shot, z, lvl, R_at_Zmin, Rmax)
            (branch == 0 && z < Z_at_Rmax) && (branch = 1)
        elseif θ == halfpi
            r = R_at_Zmin
            branch = 2
        elseif (θ > halfpi) && (θ <= π)
            # lower left
            r = find_r(shot, z, lvl, Rmin, R_at_Zmin)
            branch = (z <= Z_at_Rmin) ? 2 : 3
            #(branch == 2 && z > Z_at_Rmin) && (branch = 3)
        elseif θ < 3 * halfpi
            # upper left
            r = find_r(shot, z, lvl, Rmin, R_at_Zmax)
            branch = (z <= Z_at_Rmin) ? 2 : 3
        elseif θ == 3 * halfpi
            r = R_at_Zmax
            branch = 4
        elseif (θ > 3 * halfpi) && (θ <= twopi)
            # upper right
            r = find_r(shot, z, lvl, R_at_Zmax, Rmax)
            branch = (z > Z_at_Rmax) ? 4 : 5
        end

        aa = (r - R0) / a
        aa = max(-1, min(1, aa))
        θr = acos(aa)#(r - R0) / a)
        if branch == 0
            θr = -θr
        elseif branch == 1 || branch == 2
            θr = θr
        elseif branch == 3 || branch == 4
            θr = twopi - θr
        elseif branch == 5
            θr = twopi + θr
        end
        Fi[k] = θr - θ
    end

end

function fit_MXH!(flat, shot, lvl, Ψaxis, Raxis, Zaxis, ρaxis, Fi, Fo, P)

    Rmax, Z_at_Rmax, Rmin, Z_at_Rmin, R_at_Zmax, Zmax, R_at_Zmin, Zmin  = find_extrema(shot, lvl, Ψaxis, Raxis, Zaxis, ρaxis; algorithm = :LD_SLSQP)

    res = residual_extrema(shot, lvl, Rmax, Z_at_Rmax, Rmin, Z_at_Rmin, R_at_Zmax, Zmax, R_at_Zmin, Zmin)

    if !valid_extrema(Rmax, Z_at_Rmax, Rmin, Z_at_Rmin, R_at_Zmax, Zmax, R_at_Zmin, Zmin)
        Rmax, Z_at_Rmax, Rmin, Z_at_Rmin, R_at_Zmax, Zmax, R_at_Zmin, Zmin = find_extrema_RZ(shot, lvl, Raxis, Zaxis)
    elseif res > 1e-6
        Rmax2, Z_at_Rmax2, Rmin2, Z_at_Rmin2, R_at_Zmax2, Zmax2, R_at_Zmin2, Zmin2 = find_extrema_RZ(shot, lvl, Raxis, Zaxis)
        if abs(shot(Rmax2, Z_at_Rmax2) - lvl) < abs(shot(Rmax, Z_at_Rmax) - lvl)
            Rmax, Z_at_Rmax = Rmax2, Z_at_Rmax2
        end
        if abs(shot(Rmin2, Z_at_Rmin2) - lvl) < abs(shot(Rmin, Z_at_Rmin) - lvl)
            Rmin, Z_at_Rmin  = Rmin2, Z_at_Rmin2
        end
        if abs(shot(R_at_Zmax2, Zmax2) - lvl) < abs(shot(R_at_Zmax, Zmax) - lvl)
            R_at_Zmax, Zmax = R_at_Zmax2, Zmax2
        end
        if abs(shot(R_at_Zmin2, Zmin2) - lvl) < abs(shot(R_at_Zmin, Zmin) - lvl)
            R_at_Zmin, Zmin = R_at_Zmin2, Zmin2
        end
    end

    @assert valid_extrema(Rmax, Z_at_Rmax, Rmin, Z_at_Rmin, R_at_Zmax, Zmax, R_at_Zmin, Zmin)

    R0 = 0.5 * (Rmax + Rmin)
    a  = 0.5 * (Rmax - Rmin)
    ϵ  = a / R0
    Z0 = 0.5 * (Zmax + Zmin)
    b  = 0.5 * (Zmax - Zmin)
    κ  = b / a

    M = length(shot.cfe)
    invM2 = 1.0 / (M + 2)
    θs = range(0, twopi, 2M+5)[1:end-1]

    compute_thetar!(Fi, θs, shot, lvl, R0, Z0, a, b, Rmax, Z_at_Rmax, Rmin, Z_at_Rmin, R_at_Zmax, R_at_Zmin)

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

function update_shot!(shot::Shot, surfaces, Ψaxis, flat_δ2=nothing, flat_δ3=nothing)
    shot.surfaces .= surfaces
    shot.R0fe, shot.Z0fe, shot.ϵfe, shot.κfe, shot.c0fe, shot.cfe, shot.sfe = surfaces_FE(shot.ρ, surfaces, flat_δ2, flat_δ3)
    shot.C .= 0.0
    shot.C[2:2:end, 1] .= Ψaxis .* (1.0 .- shot.ρ.^2)
    shot.C[1:2:end, 1] .= -2.0 .* Ψaxis .* shot.ρ
    MXH_quadrature!(shot)
    metrics_quadrature!(shot)
    return shot
end

function refit!(shot::Shot, Ψaxis::Real, Raxis::Real, Zaxis::Real; debug::Bool=false, fit_fallback::Bool=true,)
    # Using ρ = rho poloidal (sqrt((Ψ-Ψaxis)/sqrt(Ψbnd-Ψaxis)))
    local surfaces, flat_δ2, flat_δ3
    warn_concentric = false
    try
        surfaces, flat_δ2, flat_δ3 = fitted_surfaces(shot, Ψaxis, Raxis, Zaxis)
    catch err
       (isa(err, InterruptException) || !fit_fallback) && rethrow(err)
       warn_concentric = true
       if debug
           println("    Warning: Fit fell back to concentric surfaces due to ", typeof(err))
       end
       surfaces = concentric_surfaces(shot, Raxis, Zaxis)
       flat_δ2 = nothing
       flat_δ3 = nothing
    end
    update_shot!(shot, surfaces, Ψaxis, flat_δ2, flat_δ3)
    return shot, warn_concentric
end

function fitted_surfaces(shot, Ψaxis, Raxis, Zaxis)
    L = length(shot.cfe)
    Fis, _, Fos, Ps = fft_prealloc_threaded(L)

    surfaces = deepcopy(shot.surfaces)

    ρaxis, _ = ρθ_RZ(shot, Raxis, Zaxis)
    #Threads.@threads
    # multithreading is broken here somehow. Maybe the tid pattern?
    for k in 2:(shot.N - 1)
        lvl = Ψaxis * (1.0 - shot.ρ[k] ^ 2)
        tid = Threads.threadid()
        Fi = Fis[tid]
        Fo = Fos[tid]
        P  = Ps[tid]
        @views flat = surfaces[:, k]
        fit_MXH!(flat, shot, lvl, Ψaxis, Raxis, Zaxis, ρaxis, Fi, Fo, P)
    end

    # Now fit two more surfaces in the last grid region so we can update derivatives
    tid = Threads.threadid()
    δρ = shot.ρ[end] - shot.ρ[end-1]

    flat_δ2 = zeros(2L+5)
    ρ_δ2 = shot.ρ[end-1] + δ_frac_2 * δρ
    lvl = Ψaxis * (1.0 - ρ_δ2 ^ 2)
    fit_MXH!(flat_δ2, shot, lvl, Ψaxis, Raxis, Zaxis, ρaxis, Fis[tid], Fos[tid], Ps[tid])

    flat_δ3  = zeros(2L+5)
    ρ_δ3  = shot.ρ[end-1] + δ_frac_3 * δρ
    lvl = Ψaxis * (1.0 - ρ_δ3 ^ 2)
    fit_MXH!(flat_δ3, shot, lvl, Ψaxis, Raxis, Zaxis, ρaxis, Fis[tid], Fos[tid], Ps[tid])


    # Extrapolate or set to zero on-axis
    ρ2 = shot.ρ[2]
    ρ3 = shot.ρ[3]
    h = 1.0 / (ρ3 - ρ2)
    surfaces[1, 1] = h .* (surfaces[1, 2] .* ρ3 .- surfaces[1, 3] .* ρ2)
    surfaces[2, 1] = h .* (surfaces[2, 2] .* ρ3 .- surfaces[2, 3] .* ρ2)
    surfaces[3, 1] = 0.0
    surfaces[4, 1] = h .* (surfaces[4, 2] .* ρ3 .- surfaces[4, 3] .* ρ2)
    surfaces[5, 1] = h .* (surfaces[5, 2] .* ρ3 .- surfaces[5, 3] .* ρ2)
    @views surfaces[6:end, 1] .= 0.0
    return surfaces, flat_δ2, flat_δ3
end


function refit_concentric!(shot::Shot, Ψaxis::Real, Raxis::Real, Zaxis::Real)
    surfaces = concentric_surfaces(shot, Raxis, Zaxis)
    update_shot!(shot, surfaces, Ψaxis)
    return shot
end


function concentric_surfaces(shot::Shot, Raxis::Real, Zaxis::Real)
    surfaces = deepcopy(shot.surfaces)
    @views boundary = shot.surfaces[:,end]
    for k in eachindex(shot.ρ)
        @views concentric_surface!(surfaces[:, k], shot.ρ[k], boundary; Raxis, Zaxis)
    end
    return surfaces
end