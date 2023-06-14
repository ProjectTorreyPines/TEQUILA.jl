function solve(shot::Shot, its::Integer;
               P=nothing, dP_dψ=nothing, F_dF_dψ=nothing, Jt_R=nothing, Jt=nothing,
               Pbnd=shot.Pbnd, Fbnd=shot.Fbnd, debug=false)
    refill = Shot(shot; P, dP_dψ, F_dF_dψ, Jt_R, Jt, Pbnd, Fbnd)
    return solve!(refill, its; debug)
end

function solve!(refill::Shot, its::Integer; debug=false)
    Fis, dFis, Fos, Ps = fft_prealloc_threaded(refill.M)
    A = preallocate_Astar(refill)
    L = 2 * refill.N * (2 * refill.M + 1)
    B = zeros(L)
    C = zeros(L)
    Ψold = 0.0
    if debug
        _, _, Ψold = find_axis(refill)
    end
    for i in 1:its
        debug && println("ITERATION $i")
        if refill.Ip_target !== nothing
            scale_Ip!(refill)
        end
        define_Astar!(A, refill, Fis, dFis, Fos, Ps)
        define_B!(B, refill, Fis[1], Fos[1], Ps[1])
        set_bc!(refill, A, B)
        C = A \ B
        refill.C .= transpose(reshape(C, (2*refill.M + 1, 2*refill.N)))
        refill.C[end, :] .= 0.0 #ensure psi=0 on boundary
        Raxis, Zaxis, Ψaxis = find_axis(refill)

        if i == 1 && its != 1
            debug && println("    Concentric surfaces used for first iteration")
            refill = refit_concentric(refill, Raxis, Zaxis)
        else
            (debug && i == 1) && println("    Trying full refit for first iteration")
            try
                refill = refit(refill, Ψaxis, Raxis, Zaxis)
            catch err
                isa(err, InterruptException) && rethrow(err)
                println("    Warning: Fell back to concentric surfaces due to ", typeof(err))
                refill = refit_concentric(refill, Raxis, Zaxis)
            end
        end

        if debug
            println("    Status: Ψaxis = $Ψaxis, Error: ", abs((Ψaxis-Ψold)/Ψaxis))
            Ψold = Ψaxis
        end
    end
    return refill
end

function scale_Ip!(shot)

    (shot.Ip_target === nothing) && return

    I_c = Ip(shot)

    if shot.Jt_R !== nothing
        Jt_R = deepcopy(shot.Jt_R)
        Jt_R.coeffs  .*= shot.Ip_target / I_c
        shot.Jt_R = Jt_R
    elseif shot.Jt !== nothing
        Jt = deepcopy(shot.Jt)
        Jt.coeffs  .*= shot.Ip_target / I_c
        shot.Jt = Jt
    else
        ΔI = shot.Ip_target - I_c
        If_c = Ip_ffp(shot)
        f = 1 + ΔI / If_c
        F_dF_dψ = deepcopy(shot.F_dF_dψ)
        F_dF_dψ.coeffs  .*= f
        shot.F_dF_dψ = F_dF_dψ
    end
    return
end