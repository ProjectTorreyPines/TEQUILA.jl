function solve(shot, its)
    shotit = deepcopy(shot)
    Fi, dFi, Fo, P = fft_prealloc(shotit.M)
    A = preallocate_Astar(shotit)
    L = 2 * shotit.N * (2 * shotit.M + 1)
    B = zeros(L)
    C = zeros(L)
    #set_bc!(shotit, A, B)
    #Alu = lu(A)
    for i in 1:its
        define_Astar!(A, shotit, Fi, dFi, Fo, P)
        define_B!(B, shotit, Fi, Fo, P)
        set_bc!(shotit, A, B)
        #Alu = i==1 ? lu(A) : Alu
        #lu!(Alu, A)
        #ldiv!(C, Alu, B)
        C = A \ B
        shotit.C .= transpose(reshape(C, (2*shotit.M + 1, 2*shotit.N)))
        _, _, Ψaxis = find_axis(shotit)
        # Define some approximate Psi contours
        levels = Ψaxis .* (1.0 .- shotit.ρ.^2)
        shotit = refit(shotit, levels)
    end
    return shotit
end;