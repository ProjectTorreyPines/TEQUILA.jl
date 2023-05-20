function solve(shot::Shot, its::Integer;
               dp_dψ=nothing, f_df_dψ=nothing, Jt_R=nothing, pbnd=nothing, fbnd=nothing, debug=false)

    pprime = dp_dψ  !== nothing ? dp_dψ   : deepcopy(shot.dp_dψ)

    ffprim = nothing
    jtor   = nothing
    if f_df_dψ !== nothing && Jt_r !== nothing
        throw(ErrorException("Must specify only one of the following: f_df_dψ, Jt_R"))
    elseif f_df_dψ !== nothing
        ffprim = f_df_dψ
        jtor = nothing
    elseif Jt_R !== nothing
        ffprim = nothing
        jtor = Jt_R
    else
        ffprim = deepcopy(shot.f_df_dψ)
        jtor = deepcopy(shot.Jt_R)
    end

    pb  = pbnd !== nothing ? pbnd : shot.pbnd
    fb  = fbnd !== nothing ? fbnd : shot.fbnd

    @assert pprime !== nothing
    @assert (ffprim !== nothing) ⊻ (jtor !== nothing)
    @assert pb !== nothing
    @assert fb !== nothing

    refill = Shot(shot; dp_dψ = pprime, f_df_dψ = ffprim, Jt_R = jtor, pbnd = pb, fbnd = fb)
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
        define_Astar!(A, refill, Fis, dFis, Fos, Ps)
        define_B!(B, refill, Fis[1], Fos[1], Ps[1])
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