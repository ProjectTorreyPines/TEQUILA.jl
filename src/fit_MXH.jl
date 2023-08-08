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

    model = Model(NLopt.Optimizer)
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

    model = Model(NLopt.Optimizer)
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

function refit(shot::Shot)
    Raxis, Zaxis, Ψaxis = find_axis(refill)
    return refit(shot, Ψaxis, Raxis, Zaxis)
end

function refit(shot::Shot, Ψaxis::Real, Raxis::Real, Zaxis::Real)

    L = length(shot.cfe)

    # Using ρ = rho poloidal (sqrt((Ψ-Ψaxis)/sqrt(Ψbnd-Ψaxis)))
    lvls = Ψaxis .* (1.0 .- shot.ρ.^2)

    Fis, _, Fos, Ps = fft_prealloc_threaded(L)

    surfaces = deepcopy(shot.surfaces)

    ρaxis, _ = ρθ_RZ(shot, Raxis, Zaxis)
    Threads.@threads for (k, lvl) in @view collect(enumerate(lvls))[2:end-1]
        tid = Threads.threadid()
        Fi = Fis[tid]
        Fo = Fos[tid]
        P  = Ps[tid]

        @views flat = surfaces[:, k]

        fit_MXH!(flat, shot, lvl, Ψaxis, Raxis, Zaxis, ρaxis, Fi, Fo, P)
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

    shot_refit = Shot(shot.N, shot.M, shot.ρ, surfaces, shot;
                      P = shot.P, dP_dψ = shot.dP_dψ,
                      F_dF_dψ = shot.F_dF_dψ, Jt_R = shot.Jt_R, Jt = shot.Jt,
                      Pbnd = shot.Pbnd, Fbnd = shot.Fbnd, Ip_target = shot.Ip_target)

    return shot_refit

end

function refit_concentric(shot::Shot, Raxis::Real, Zaxis::Real)

    surfaces = deepcopy(shot.surfaces)
    @views boundary = shot.surfaces[:,end]
    for k in eachindex(shot.ρ)
        @views concentric_surface!(surfaces[:, k], shot.ρ[k], boundary; Raxis, Zaxis)
    end

    shot_refit = Shot(shot.N, shot.M, shot.ρ, surfaces, shot;
                      P = shot.P, dP_dψ = shot.dP_dψ,
                      F_dF_dψ = shot.F_dF_dψ, Jt_R = shot.Jt_R, Jt = shot.Jt,
                      Pbnd = shot.Pbnd, Fbnd = shot.Fbnd, Ip_target = shot.Ip_target)

    return shot_refit

end

function refit_concentric(shot::Shot)

    surfaces = deepcopy(shot.surfaces)
    @views boundary = shot.surfaces[:,end]
    for k in eachindex(shot.ρ)
        @views concentric_surface!(surfaces[:, k], shot.ρ[k], boundary)
    end

    shot_refit = Shot(shot.N, shot.M, shot.ρ, surfaces, shot;
                      P = shot.P, dP_dψ = shot.dP_dψ,
                      F_dF_dψ = shot.F_dF_dψ, Jt_R = shot.Jt_R, Jt = shot.Jt,
                      Pbnd = shot.Pbnd, Fbnd = shot.Fbnd, Ip_target = shot.Ip_target)

    return shot_refit

end

function refit_shifted(shot::Shot, Raxis::Real, Zaxis::Real)
    surfaces = deepcopy(shot.surfaces)
    @views boundary = shot.surfaces[:,end]
    Rcntr = surfaces[1, 1]
    Zcntr = surfaces[2, 1]
    Rint, Zint = TEQUILA.boundary_intersection(boundary, Rcntr, Zcntr, Raxis, Zaxis)
    fac = sqrt(((Raxis - Rint) ^ 2 + (Zaxis - Zint) ^ 2)/((Rcntr - Rint) ^ 2 + (Zcntr - Zint) ^ 2))
    for k in eachindex(shot.ρ)
        (k == 1 || k == shot.N) && continue
        @views TEQUILA.shift_surface!(surfaces[:, k], fac, Rcntr, Zcntr, Raxis, Zaxis, Rint, Zint)
    end
    surfaces[1, 1] = Raxis
    surfaces[2, 1] = Zaxis

    shot_refit = Shot(shot.N, shot.M, shot.ρ, surfaces, shot;
                      P = shot.P, dP_dψ = shot.dP_dψ,
                      F_dF_dψ = shot.F_dF_dψ, Jt_R = shot.Jt_R, Jt = shot.Jt,
                      Pbnd = shot.Pbnd, Fbnd = shot.Fbnd, Ip_target = shot.Ip_target)

    return shot_refit

end

function find_ρ_lvls(θ, shot, lvls; pad=0)
    ρ_lvls = similar(lvls)
    Nx = shot.N + pad
    x=zeros(Nx)
    coeffs=zeros(2Nx)
    return find_ρ_lvls!(ρ_lvls, θ, shot, lvls; pad, x, coeffs)
end

function find_ρ_lvls!(ρ_lvls, θ, shot, lvls; pad=0, x=zeros(shot.N+pad), coeffs=zeros(2*(shot.N+pad)))

    # first x is the ρ coordinate
    # pad near x=0 to better resolve steep gradient
    @views x[(2+pad):end] .= shot.ρ[2:end]
    for j in 1:pad
        k = 2 + pad - j
        x[k] = x[k+1] / 2
    end
    x[1] = 0.0

    coeffs[1:2:end] .= 1.0 ./ TEQUILA.dpsi_dρ.(Ref(shot), x, θ)
    coeffs[2:2:end] .= x

    x2 = x[2]

    # reuse x for psi to avoid allocation
    x .= psi_ρθ.(Ref(shot), x, θ)

    coeffs[1] = (2 * x2 / (x[2] - x[1])) - coeffs[3] # quadratic through first grid cell to avoid infinite gradient
    f = FE_rep(x, coeffs)
    ρ_lvls .= f.(lvls)
    ρ_lvls[1] = 0.0
    return ρ_lvls
end

function ρθ_contours(shot, lvls; pad=4)
    Nlvls = length(lvls)
    L, _ = size(shot.surfaces)
    L = (L - 5) ÷ 2
    M = 4 * (shot.M + L)
    Rs = zeros(M, Nlvls)
    Zs = zeros(M, Nlvls)
    ρ_lvls = similar(lvls)
    return ρθ_contours!(Rs, Zs, shot, lvls; pad, ρ_lvls)
end

function ρθ_contours!(Rs, Zs, shot, lvls; pad=4, ρ_lvls = similar(lvls))
    M, _ = size(Rs)
    θs = range(0, 2π, M+1)[1:end-1]
    Nx = shot.N + pad
    x = zeros(Nx)
    coeffs = zeros(2Nx)
    for (k, θ) in enumerate(θs)
        find_ρ_lvls!(ρ_lvls, θ, shot, lvls; pad, x, coeffs)
        for (j, ρ) in enumerate(ρ_lvls)
            Rs[k, j], Zs[k, j] = R_Z(shot, ρ, θ)
        end
    end
    return Rs, Zs
end

function refit2(shot::Shot)
    Raxis, Zaxis, Ψaxis = find_axis(refill)
    return refit2(shot, Ψaxis, Raxis, Zaxis)
end

function refit2(shot::Shot, Ψaxis::Real, Raxis::Real, Zaxis::Real)

    mid = refit_shifted(shot, Raxis, Zaxis)
    lvls = Ψaxis .* (1.0 .- mid.ρ.^2)
    Rs, Zs = ρθ_contours(mid, lvls)

    surfaces = deepcopy(mid.surfaces)
    L, N = size(mid.surfaces)
    L = (L - 5) ÷ 2
    Stmp = MXH(Raxis, L)

    M, _ = size(Rs)
    θ   = zeros(M)
    Δθᵣ = zeros(M)
    dθ  = zeros(M)
    Fm  = zeros(M)

    for j in eachindex(mid.ρ)
        (j == 1 || j == N) && continue
        @views pr = Rs[:, j]
        @views pz = Zs[:, j]
        MXH!(Stmp, pr, pz; θ, Δθᵣ, dθ, Fm, spline=true)
        @views flat_coeffs!(surfaces[:, j], Stmp)
    end
    surfaces[1, 1] = Raxis
    surfaces[2, 1] = Zaxis

    shot_refit = Shot(mid.N, mid.M, mid.ρ, surfaces, mid;
                      P = mid.P, dP_dψ = mid.dP_dψ,
                      F_dF_dψ = mid.F_dF_dψ, Jt_R = mid.Jt_R, Jt = mid.Jt,
                      Pbnd = mid.Pbnd, Fbnd = mid.Fbnd, Ip_target = mid.Ip_target)

end