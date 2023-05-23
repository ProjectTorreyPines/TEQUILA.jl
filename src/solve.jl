function solve(shot::Shot, its::Integer;
               P=nothing, dP_dψ=nothing, F_dF_dψ=nothing, Jt_R=nothing, Jt=nothing,
               Pbnd=shot.Pbnd, Fbnd=shot.Fbnd, debug=false)
    refill = Shot(shot; P, dP_dψ, F_dF_dψ, Jt_R, Jt, Pbnd, Fbnd)
    return solve!(refill, its; debug)
end

function solve!(refill::Shot, its::Integer; debug=false)
    Fi, dFi, Fo, P = fft_prealloc(refill.M)
    A = preallocate_Astar(refill)
    L = 2 * refill.N * (2 * refill.M + 1)
    B = zeros(L)
    C = zeros(L)
    Ψold = 0.0
    if debug
        _, _, Ψold = find_axis(refill)
    end
    for i in 1:its
        define_Astar!(A, refill, Fi, dFi, Fo, P)
        define_B!(B, refill, Fi, Fo, P)
        set_bc!(refill, A, B)

        C = A \ B
        refill.C .= transpose(reshape(C, (2*refill.M + 1, 2*refill.N)))
        refill.C[end, :] .= 0.0 #ensure psi=0 on boundary
        _, _, Ψaxis = find_axis(refill)

        # Using ρ = rho poloidal (sqrt((Ψ-Ψaxis)/sqrt(Ψbnd-Ψaxis)))
        levels = Ψaxis .* (1.0 .- refill.ρ.^2)
        #levels = Ψaxis .* (1.0 .- refill.ρ) .^0.5

        refill = refit(refill, levels)
        if debug
            println("Iteration ", i, ": Ψaxis = ", Ψaxis, ", Error: ", abs(Ψaxis-Ψold))
            Ψold = Ψaxis
        end
    end
    return refill
end