"""
    solve(
        shot::Shot,
        its::Integer;
        tol::Real=0.0,
        relax::Real=1.0,
        debug::Bool=false,
        fit_fallback::Bool=true,
        concentric_first::Bool=true,
        P::Union{Nothing,Tuple{<:Union{FE_rep,Function},Symbol},Profile}=nothing,
        dP_dψ::Union{Nothing,Tuple{<:Union{FE_rep,Function},Symbol},Profile}=nothing,
        F_dF_dψ::Union{Nothing,Tuple{<:Union{FE_rep,Function},Symbol},Profile}=nothing,
        Jt_R::Union{Nothing,Tuple{<:Union{FE_rep,Function},Symbol},Profile}=nothing,
        Jt::Union{Nothing,Tuple{<:Union{FE_rep,Function},Symbol},Profile}=nothing,
        Pbnd=shot.Pbnd,
        Fbnd=shot.Fbnd,
        Ip_target=shot.Ip_target
    )

Solve the equilibrium, initially defined by `shot` with `its` iterations.
Pressure and current information taken from `shot` unless provided in keywords

Returns a new Shot, often called `refill` by convention

# Keyword arguments

  - `tol` - Relative tolerance for convergence of the magnetic axis flux value to terminate iterations early
  - `relax` - Relaxation parameter on the Picard iterations. `Ψₙ₊₁ = relax * Ψ̃ₙ₊₁ + (1 - relax) * Ψₙ`
  - `debug=true` - Print debugging and convergence information
  - `fit_fallback=true` - Use concentric surfaces if any flux surface errors on refitting. Improves robustness in early iterations
  - `concentric_first=true` - Use concentric surfaces for first iteration, which can improve robustness if large changes from `shot` is expected
"""
function solve(
    shot::Shot,
    its::Integer;
    tol::Real=0.0,
    relax::Real=1.0,
    debug::Bool=false,
    fit_fallback::Bool=true,
    concentric_first::Bool=true,
    P::ProfType=nothing,
    dP_dψ::ProfType=nothing,
    F_dF_dψ::ProfType=nothing,
    Jt_R::ProfType=nothing,
    Jt::ProfType=nothing,
    Pbnd=shot.Pbnd,
    Fbnd=shot.Fbnd,
    Ip_target=shot.Ip_target
)
    refill = Shot(shot; P, dP_dψ, F_dF_dψ, Jt_R, Jt, Pbnd, Fbnd, Ip_target)
    return solve!(refill, its; tol, relax, debug, fit_fallback, concentric_first)
end

function solve!(refill::Shot, its::Integer; tol::Real=0.0, relax::Real=1.0, debug::Bool=false, fit_fallback::Bool=true, concentric_first::Bool=true)
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
            refill.C .= transpose(reshape(C, (2 * refill.M + 1, 2 * refill.N)))
        else
            refill.C .= (1.0 - relax) .* refill.C .+ relax .* transpose(reshape(C, (2 * refill.M + 1, 2 * refill.N)))
        end
        refill.C[end, :] .= 0.0 #ensure psi=0 on boundary

        Raxis, Zaxis, Ψaxis = find_axis(refill)

        if concentric_first && i == 1
            debug && println("    Concentric surfaces used for first iteration")
            refill = refit_concentric!(refill, Ψaxis, Raxis, Zaxis)
        else
            refill, warn_concentric = refit!(refill, Ψaxis, Raxis, Zaxis; debug, fit_fallback)
        end

        error = abs((Ψaxis - Ψold) / Ψaxis)
        debug && println("    Status: Ψaxis = $Ψaxis, Error: $error")
        Ψold = Ψaxis
        if error <= tol && i > 1 && !warn_concentric
            debug && println("DONE: Successful convergence")
            break
        end

        if i == its
            debug && println("DONE: maximum iterations")
            break
        end

    end
    warn_concentric && @warn("Final iteration used concentric surfaces and is likely inaccurate")
    return refill
end

function scale_Ip!(shot::Shot, I_c=Ip(shot))
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

function validate_current(shot; I_c=Ip(shot))
    sign_Ip = sign(I_c)
    error_text(name, values) =
        "Provided $name profile produces regions with current opposite to the total current ($(sign_Ip)).\nNot allowed since Ψ becomes nonmonotonic - Please correct input profile\n($values    )"
    if shot.Jt_R !== nothing
        signs = sign.(shot.Jt_R.(shot.ρ))
        @assert all(s ∈ (sign_Ip, 0.0) for s in signs) error_text("Jt_R", shot.Jt_R.(shot.ρ))
    elseif shot.Jt !== nothing
        signs = sign.(shot.Jt.(shot.ρ))
        @assert all(s ∈ (sign_Ip, 0.0) for s in signs) error_text("Jt", shot.Jt.(shot.ρ))
    else
        invR2 = FE_rep(shot, fsa_invR2)
        Pp = Pprime(shot, shot.P, shot.dP_dψ)
        @assert all(sign(-(Pp(x) + invR2(x) * shot.F_dF_dψ(x) / μ₀)) ∈ (sign_Ip, 0.0) for x in shot.ρ) error_text("F_dF_dψ")
    end
    return
end