function solve(shot::Shot, its::Integer; tol::Real=0.0, relax::Real = 1.0,
               debug::Bool=false, fit_fallback::Bool=true, concentric_first::Bool=true,
               profile_grid=:poloidal, P=nothing, dP_dψ=nothing, F_dF_dψ=nothing, Jt_R=nothing, Jt=nothing,
               Pbnd=shot.Pbnd, Fbnd=shot.Fbnd, Ip_target=shot.Ip_target)
    refill = Shot(shot; profile_grid, P, dP_dψ, F_dF_dψ, Jt_R, Jt, Pbnd, Fbnd, Ip_target)
    return solve!(refill, its; tol, relax, debug, fit_fallback, concentric_first)
end

function solve!(refill::Shot, its::Integer; tol::Real=0.0, relax::Real=1.0,
                debug::Bool=false, fit_fallback::Bool=true, concentric_first::Bool=true)

    if debug
        pstr = (refill.P !== nothing) ? "P on $(refill.P.grid) grid" : "dP_dψ on $(refill.dP_dψ.grid) grid"
        if refill.F_dF_dψ !== nothing
            jstr = "F_dF_dψ on $(refill.F_dF_dψ.grid) grid"
        elseif refill.Jt_R !== nothing
            jstr = "Jt_R on $(refill.Jt_R.grid) grid"
        else
            jstr = "Jt on $(refill.Jt.grid) grid"
        end
        println("*** Solving equilibrium with " * pstr * " and " * jstr * " ***")
    end

    # validate current
    I_c = Ip(refill)
    validate_current(refill; I_c)

    Fis, dFis, Fos, Ps = fft_prealloc_threaded(refill.M)
    A = preallocate_Astar(refill)
    L = 2 * refill.N * (2 * refill.M + 1)
    B = zeros(L)
    C = zeros(L)
    Ψold = 0.0
    warn_concentric = false
    _, _, Ψold = find_axis(refill)

    local linsolve
    for i in 1:its
        debug && println("ITERATION $i")

        # move to rho_tor grid and scale current, if necessary
        update_profiles!(refill)
        scale_Ip!(refill)

        define_Astar!(A, refill, Fis, dFis, Fos, Ps)
        define_B!(B, refill, Fis, Fos, Ps)
        set_bc!(refill, A, B)

        if i == 1
            prob = LinearProblem(A, B)
            linsolve = LinearSolve.init(prob)
        else
            linsolve.A = A
            linsolve.b = B
        end
        sol = LinearSolve.solve!(linsolve)
        C = sol.u

        if i == 1
            refill.C .= transpose(reshape(C, (2*refill.M + 1, 2*refill.N)))
        else
            refill.C .= (1.0-relax) .* refill.C .+ relax .* transpose(reshape(C, (2*refill.M + 1, 2*refill.N)))
        end
        refill.C[end, :] .= 0.0 #ensure psi=0 on boundary

        Raxis, Zaxis, Ψaxis = find_axis(refill)

        if concentric_first && i == 1
            debug && println("    Concentric surfaces used for first iteration")
            refill = refit_concentric!(refill, Ψaxis, Raxis, Zaxis)
        else
            refill, warn_concentric = refit!(refill, Ψaxis, Raxis, Zaxis; debug, fit_fallback)
        end

        error = abs((Ψaxis-Ψold)/Ψaxis)
        debug && println("    Status: Ψaxis = $Ψaxis, Error: $error")
        Ψold = Ψaxis
        if error <= tol && i > 1
            debug && println("DONE: Successful convergence")
            break
        end

        if i == its
            debug && println("DONE: maximum iterations")
            break
        end

    end
    warn_concentric && println("WARNING: Final iteration used concentric surfaces and is likely inaccurate")
    return refill
end

function scale_Ip!(shot::Shot, I_c = Ip(shot))

    (shot.Ip_target === nothing) && return

    if shot.Jt_R !== nothing
        Jt_R = deepcopy(shot.Jt_R)
        Jt_R.fe.coeffs .*= shot.Ip_target / I_c
        shot.Jt_R = Jt_R
    elseif shot.Jt !== nothing
        Jt = deepcopy(shot.Jt)
        Jt.fe.coeffs .*= shot.Ip_target / I_c
        shot.Jt = Jt
    else
        ΔI = shot.Ip_target - I_c
        If_c = Ip_ffp(shot)
        fac = 1 + ΔI / If_c
        F_dF_dψ = deepcopy(shot.F_dF_dψ)
        F_dF_dψ.fe.coeffs .*= fac
        shot.F_dF_dψ = F_dF_dψ
    end
    return
end

function validate_current(shot; I_c = Ip(shot))
    sign_Ip = sign(I_c)
    if shot.Jt_R !== nothing
        good_sign = (sign(shot.Jt_R(x)) != -sign_Ip for x in shot.ρ)
    elseif shot.Jt !== nothing
        good_sign = (sign(shot.Jt(x)) != -sign_Ip for x in shot.ρ)
    else
        invR2 = FE_rep(shot, fsa_invR2)
        Pp = Pprime(shot, shot.P, shot.dP_dψ)
        good_sign = (sign(-(Pp(x) + invR2(x) * shot.F_dF_dψ(x) / μ₀)) != -sign_Ip for x in shot.ρ)
    end
    if !all(good_sign)
        throw(ErrorException("Provided F_dF_dψ, Jt, or Jt_R profile produces regions with current opposite total current\n"*
                             "       Not allowed since Ψ becomes nonmonotonic - Please correct input profile"))
    end
    return
end