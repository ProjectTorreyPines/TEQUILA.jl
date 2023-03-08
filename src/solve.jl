function solve(shot, dp_dψ, f_df_dψ, its)
    shotit = deepcopy(shot)
    Fi, dFi, Fo, P = fft_prealloc(shotit.M)
    A = preallocate_Astar(shotit)
    L = 2 * shotit.N * (2 * shotit.M + 1)
    B = zeros(L)
    C = zeros(L)
    
    for i in 1:its
        define_Astar!(A, shotit, Fi, dFi, Fo, P)
        define_B!(B, shotit, dp_dψ, f_df_dψ, Fi, Fo, P)
        set_bc!(shotit, A, B)
        
        C = A \ B
        shotit.C .= transpose(reshape(C, (2*shotit.M + 1, 2*shotit.N)))
        _, _, Ψaxis = find_axis(shotit)
       
        # Using ρ = rho poloidal (sqrt((Ψ-Ψaxis)/sqrt(Ψbnd-Ψaxis)))
        levels = Ψaxis .* (1.0 .- shotit.ρ.^2)
        #levels = Ψaxis .* (1.0 .- shotit.ρ) .^0.5
        
        shotit = refit(shotit, levels)
    end
    return shotit
end;