b2e(shot::Shot, j::Integer; m::Integer = 1) = (2 * shot.M + 1) * (j - 1) + m

function predefine_block!(X, Y, I, J, M, block)
    start = (block - 1) * (2M + 1)^2 + 1
    for l in 0:2M
        for k in 0:2M
            n = start + (2M + 1) * l + k
            X[n] = I + l
            Y[n] = J + k
        end
    end
    return
end

function preallocate_Astar(shot)

    M = shot.M
    N = shot.N

    # total number of nonzeros
    L = (2M + 1)^2 * (12N - 8)
    X = Vector{Float64}(undef, L)
    Y = Vector{Float64}(undef, L)
    V = Vector{Float64}(undef, L)
    V .= NaN

    block = 1

    for j in 1:N

        # block column
        je = 2j
        jo = je - 1

        # element column
        Jo = b2e(shot, jo)
        Je = b2e(shot, je)

        if j > 1
            # [jo, jo-3] block does not exist

            # [je, je-3]
            Ie = b2e(shot, je-3)
            predefine_block!(X, Y, Ie, Je, M, block)
            block += 1

            # T[jo, jo-2]
            Io = b2e(shot, jo-2)
            predefine_block!(X, Y, Io, Jo, M, block)
            block += 1

            # T[je, je-2]
            Ie = b2e(shot, je-2)
            predefine_block!(X, Y, Ie, Je, M, block)
            block += 1

            # T[jo, jo-1]
            Io = b2e(shot, jo-1)
            predefine_block!(X, Y, Io, Jo, M, block)
            block += 1
        end
        # T[je, je-1]
        Ie = b2e(shot, je-1)
        predefine_block!(X, Y, Ie, Je, M, block)
        block += 1

        # T[jo, jo]
        Io = Jo
        predefine_block!(X, Y, Io, Jo, M, block)
        block += 1

        # T[je, je]
        Ie = Je
        predefine_block!(X, Y, Ie, Je, M, block)
        block += 1

        # T[jo, jo+1]
        Io = b2e(shot, jo+1)
        predefine_block!(X, Y, Io, Jo, M, block)
        block += 1

        if j < N
            # T[je, je+1]
            Ie = b2e(shot, je+1)
            predefine_block!(X, Y, Ie, Je, M, block)
            block += 1

            # T[jo, jo+2]
            Io = b2e(shot, jo+2)
            predefine_block!(X, Y, Io, Jo, M, block)
            block += 1

            # T[je, je+2]
            Ie = b2e(shot, je+2)
            predefine_block!(X, Y, Ie, Je, M, block)
            block += 1

            # T[jo, jo+3]
            Io = b2e(shot, jo+3)
            predefine_block!(X, Y, Io, Jo, M, block)
            block += 1
            # T[je, je+3] does not exist
        end
    end
    return sparse(X, Y, V)
end

function define_Astar(shot)

    Astar = preallocate_Astar(shot)
    define_Astar!(Astar, shot)
    return Astar
end

function define_Astar!(Astar, shot)

    Fi, Fo, P = fft_prealloc(shot.M)
    define_Astar!(Astar, shot, Fi, Fo, P)
    return
end

function define_Astar!(Astar, shot, Fi, Fo, P)

    N = shot.N
    M = shot.M
    ρ = shot.ρ

    Astar.nzval .= 0.0

    mrange = 0:2M

    # Loop over columns of
    for m in 0:M

        Mc = 2m
        Ms = 2m - 1      # note that Ms = 0 for m = 0 won't exist

        # # if m != 0
        # #     cgrr  = (x, t) -> cos(m * t) * gρρ(shot, x, t)
        # #     cgrt  = (x, t) -> cos(m * t) * gρθ(shot, x, t)
        # #     dcgrt = (x, t) -> -m * sin(m * t) * gρθ(shot, x, t)
        # #     dcgtt = (x, t) -> -m * sin(m * t) * gθθ(shot, x, t)

        # #     sgrr  = (x, t) -> sin(m * t) * gρρ(shot, x, t)
        # #     sgrt  = (x, t) -> sin(m * t) * gρθ(shot, x, t)
        # #     dsgrt = (x, t) -> m * cos(m * t) * gρθ(shot, x, t)
        # #     dsgtt = (x, t) -> m * cos(m * t) * gθθ(shot, x, t)
        # # end
        # # cgrr(x, t)  = (m == 0 ? cos(t) : cos(m * t)) * gρρ(shot, x, t)
        # # cgrt(x, t)  = (m == 0 ? cos(t) : cos(m * t)) * gρθ(shot, x, t)
        # # dcgrt(x, t) = (m == 0 ? -sin(t) : -m * sin(m * t)) * gρθ(shot, x, t)
        # # dcgtt(x, t) = (m == 0 ? -sin(t) : -m * sin(m * t)) * gθθ(shot, x, t)

        # # sgrr(x, t)  = (m == 0) ? 0.0 : sin(m * t) * gρρ(shot, x, t)
        # # sgrt(x, t)  = (m == 0) ? 0.0 : sin(m * t) * gρθ(shot, x, t)
        # # dsgrt(x, t) = (m == 0) ? 0.0 : m * cos(m * t) * gρθ(shot, x, t)
        # # dsgtt(x, t) = (m == 0) ? 0.0 : m * cos(m * t) * gθθ(shot, x, t)

        cgrr(x, t)  =    -cos(m * t) * gρρ(shot, x, t)
        cgrt(x, t)  =    -cos(m * t) * gρθ(shot, x, t)
        dcgrt(x, t) = m * sin(m * t) * gρθ(shot, x, t)
        dcgtt(x, t) = m * sin(m * t) * gθθ(shot, x, t)

        sgrr(x, t)  =     -sin(m * t) * gρρ(shot, x, t)
        sgrt(x, t)  =     -sin(m * t) * gρθ(shot, x, t)
        dsgrt(x, t) = -m * cos(m * t) * gρθ(shot, x, t)
        dsgtt(x, t) = -m * cos(m * t) * gθθ(shot, x, t)

        # loop over rows of blocks
        for j in 1:N

            # block row
            je = 2j
            jo = je - 1

            # element row
            Jo = b2e(shot, jo)
            Je = b2e(shot, je)
            Jos = Jo .+ mrange
            Jes = Je .+ mrange

            if j > 1
                # [jo, jo-3] does not exist

                # [je, je-3]
                Ie = b2e(shot, je-3)
                @views θFD_ρIP_f_nu_nu!(Astar[Jes, Ie + Mc], cgrr, D_νe, j, D_νo, j-1, ρ, M, Fi, Fo, P)
                @views θFD_ρIP_f_nu_nu!(Astar[Jes, Ie + Mc], cgrt, νe, j, D_νo, j-1, ρ, M, Fi, Fo, P, fft_op=:derivative)
                @views θFD_ρIP_f_nu_nu!(Astar[Jes, Ie + Mc], dcgrt, D_νe, j, νo, j-1, ρ, M, Fi, Fo, P)
                @views θFD_ρIP_f_nu_nu!(Astar[Jes, Ie + Mc], dcgtt, νe, j, νo, j-1, ρ, M, Fi, Fo, P, fft_op = :derivative)
                if m != 0
                    @views θFD_ρIP_f_nu_nu!(Astar[Jes, Ie + Ms], sgrr, D_νe, j, D_νo, j-1, ρ, M, Fi, Fo, P)
                    @views θFD_ρIP_f_nu_nu!(Astar[Jes, Ie + Ms], sgrt, νe, j, D_νo, j-1, ρ, M, Fi, Fo, P, fft_op=:derivative)
                    @views θFD_ρIP_f_nu_nu!(Astar[Jes, Ie + Ms], dsgrt, D_νe, j, νo, j-1, ρ, M, Fi, Fo, P)
                    @views θFD_ρIP_f_nu_nu!(Astar[Jes, Ie + Ms], dsgtt, νe, j, νo, j-1, ρ, M, Fi, Fo, P, fft_op = :derivative)
                end

                # T[jo, jo-2]
                Io = b2e(shot, jo-2)
                @views θFD_ρIP_f_nu_nu!(Astar[Jos, Io + Mc], cgrr, D_νo, j, D_νo, j-1, ρ, M, Fi, Fo, P)
                @views θFD_ρIP_f_nu_nu!(Astar[Jos, Io + Mc], cgrt, νo, j, D_νo, j-1, ρ, M, Fi, Fo, P, fft_op=:derivative)
                @views θFD_ρIP_f_nu_nu!(Astar[Jos, Io + Mc], dcgrt, D_νo, j, νo, j-1, ρ, M, Fi, Fo, P)
                @views θFD_ρIP_f_nu_nu!(Astar[Jos, Io + Mc], dcgtt, νo, j, νo, j-1, ρ, M, Fi, Fo, P, fft_op = :derivative)
                if m != 0
                    @views θFD_ρIP_f_nu_nu!(Astar[Jos, Io + Ms], sgrr, D_νo, j, D_νo, j-1, ρ, M, Fi, Fo, P)
                    @views θFD_ρIP_f_nu_nu!(Astar[Jos, Io + Ms], sgrt, νo, j, D_νo, j-1, ρ, M, Fi, Fo, P, fft_op=:derivative)
                    @views θFD_ρIP_f_nu_nu!(Astar[Jos, Io + Ms], dsgrt, D_νo, j, νo, j-1, ρ, M, Fi, Fo, P)
                    @views θFD_ρIP_f_nu_nu!(Astar[Jos, Io + Ms], dsgtt, νo, j, νo, j-1, ρ, M, Fi, Fo, P, fft_op = :derivative)
                end

                # T[je, je-2]
                Ie = b2e(shot, je-2)
                @views θFD_ρIP_f_nu_nu!(Astar[Jes, Ie + Mc], cgrr, D_νe, j, D_νe, j-1, ρ, M, Fi, Fo, P)
                @views θFD_ρIP_f_nu_nu!(Astar[Jes, Ie + Mc], cgrt, νe, j, D_νe, j-1, ρ, M, Fi, Fo, P, fft_op=:derivative)
                @views θFD_ρIP_f_nu_nu!(Astar[Jes, Ie + Mc], dcgrt, D_νe, j, νe, j-1, ρ, M, Fi, Fo, P)
                @views θFD_ρIP_f_nu_nu!(Astar[Jes, Ie + Mc], dcgtt, νe, j, νe, j-1, ρ, M, Fi, Fo, P, fft_op = :derivative)
                if m != 0
                    @views θFD_ρIP_f_nu_nu!(Astar[Jes, Ie + Ms], sgrr, D_νe, j, D_νe, j-1, ρ, M, Fi, Fo, P)
                    @views θFD_ρIP_f_nu_nu!(Astar[Jes, Ie + Ms], sgrt, νe, j, D_νe, j-1, ρ, M, Fi, Fo, P, fft_op=:derivative)
                    @views θFD_ρIP_f_nu_nu!(Astar[Jes, Ie + Ms], dsgrt, D_νe, j, νe, j-1, ρ, M, Fi, Fo, P)
                    @views θFD_ρIP_f_nu_nu!(Astar[Jes, Ie + Ms], dsgtt, νe, j, νe, j-1, ρ, M, Fi, Fo, P, fft_op = :derivative)
                end

                # T[jo, jo-1]
                Io = b2e(shot, jo-1)
                @views θFD_ρIP_f_nu_nu!(Astar[Jos, Io + Mc], cgrr, D_νo, j, D_νe, j-1, ρ, M, Fi, Fo, P)
                @views θFD_ρIP_f_nu_nu!(Astar[Jos, Io + Mc], cgrt, νo, j, D_νe, j-1, ρ, M, Fi, Fo, P, fft_op=:derivative)
                @views θFD_ρIP_f_nu_nu!(Astar[Jos, Io + Mc], dcgrt, D_νo, j, νe, j-1, ρ, M, Fi, Fo, P)
                @views θFD_ρIP_f_nu_nu!(Astar[Jos, Io + Mc], dcgtt, νo, j, νe, j-1, ρ, M, Fi, Fo, P, fft_op = :derivative)
                if m != 0
                    @views θFD_ρIP_f_nu_nu!(Astar[Jos, Io + Ms], sgrr, D_νo, j, D_νe, j-1, ρ, M, Fi, Fo, P)
                    @views θFD_ρIP_f_nu_nu!(Astar[Jos, Io + Ms], sgrt, νo, j, D_νe, j-1, ρ, M, Fi, Fo, P, fft_op=:derivative)
                    @views θFD_ρIP_f_nu_nu!(Astar[Jos, Io + Ms], dsgrt, D_νo, j, νe, j-1, ρ, M, Fi, Fo, P)
                    @views θFD_ρIP_f_nu_nu!(Astar[Jos, Io + Ms], dsgtt, νo, j, νe, j-1, ρ, M, Fi, Fo, P, fft_op = :derivative)
                end
            end
            # T[je, je-1]
            Ie = b2e(shot, je-1)
            @views θFD_ρIP_f_nu_nu!(Astar[Jes, Ie + Mc], cgrr, D_νe, j, D_νo, j, ρ, M, Fi, Fo, P)
            @views θFD_ρIP_f_nu_nu!(Astar[Jes, Ie + Mc], cgrt, νe, j, D_νo, j, ρ, M, Fi, Fo, P, fft_op=:derivative)
            @views θFD_ρIP_f_nu_nu!(Astar[Jes, Ie + Mc], dcgrt, D_νe, j, νo, j, ρ, M, Fi, Fo, P)
            @views θFD_ρIP_f_nu_nu!(Astar[Jes, Ie + Mc], dcgtt, νe, j, νo, j, ρ, M, Fi, Fo, P, fft_op = :derivative)
            if m != 0
                @views θFD_ρIP_f_nu_nu!(Astar[Jes, Ie + Ms], sgrr, D_νe, j, D_νo, j, ρ, M, Fi, Fo, P)
                @views θFD_ρIP_f_nu_nu!(Astar[Jes, Ie + Ms], sgrt, νe, j, D_νo, j, ρ, M, Fi, Fo, P, fft_op=:derivative)
                @views θFD_ρIP_f_nu_nu!(Astar[Jes, Ie + Ms], dsgrt, D_νe, j, νo, j, ρ, M, Fi, Fo, P)
                @views θFD_ρIP_f_nu_nu!(Astar[Jes, Ie + Ms], dsgtt, νe, j, νo, j, ρ, M, Fi, Fo, P, fft_op = :derivative)
            end

            # Boundary term [νj gρρ D_νi]_0^1, non-zero for νj even and νi odd
            # sign flipped to account for later reversal
            if j == 1
                # ρ=0
                @views fourier_decompose!(Astar[Jes, Ie + Mc], θ ->  cgrr(0.0, θ), M, Fi, Fo, P)
                if m != 0
                    @views fourier_decompose!(Astar[Jes, Ie + Ms], θ ->  sgrr(0.0, θ), M, Fi, Fo, P)
                end
            elseif j == N
                # ρ=1
                @views fourier_decompose!(Astar[Jes, Ie + Mc], θ -> -cgrr(1.0, θ), M, Fi, Fo, P)
                if m != 0
                    @views fourier_decompose!(Astar[Jes, Ie + Ms], θ -> -sgrr(1.0, θ), M, Fi, Fo, P)
                end
            end


            # T[jo, jo]
            Io = Jo
            @views θFD_ρIP_f_nu_nu!(Astar[Jos, Io + Mc], cgrr, D_νo, j, D_νo, j, ρ, M, Fi, Fo, P)
            @views θFD_ρIP_f_nu_nu!(Astar[Jos, Io + Mc], cgrt, νo, j, D_νo, j, ρ, M, Fi, Fo, P, fft_op=:derivative)
            @views θFD_ρIP_f_nu_nu!(Astar[Jos, Io + Mc], dcgrt, D_νo, j, νo, j, ρ, M, Fi, Fo, P)
            @views θFD_ρIP_f_nu_nu!(Astar[Jos, Io + Mc], dcgtt, νo, j, νo, j, ρ, M, Fi, Fo, P, fft_op = :derivative)
            if m != 0
                @views θFD_ρIP_f_nu_nu!(Astar[Jos, Io + Ms], sgrr, D_νo, j, D_νo, j, ρ, M, Fi, Fo, P)
                @views θFD_ρIP_f_nu_nu!(Astar[Jos, Io + Ms], sgrt, νo, j, D_νo, j, ρ, M, Fi, Fo, P, fft_op=:derivative)
                @views θFD_ρIP_f_nu_nu!(Astar[Jos, Io + Ms], dsgrt, D_νo, j, νo, j, ρ, M, Fi, Fo, P)
                @views θFD_ρIP_f_nu_nu!(Astar[Jos, Io + Ms], dsgtt, νo, j, νo, j, ρ, M, Fi, Fo, P, fft_op = :derivative)
            end

            # T[je, je]
            Ie = Je
            @views θFD_ρIP_f_nu_nu!(Astar[Jes, Ie + Mc], cgrr, D_νe, j, D_νe, j, ρ, M, Fi, Fo, P)
            @views θFD_ρIP_f_nu_nu!(Astar[Jes, Ie + Mc], cgrt, νe, j, D_νe, j, ρ, M, Fi, Fo, P, fft_op=:derivative)
            @views θFD_ρIP_f_nu_nu!(Astar[Jes, Ie + Mc], dcgrt, D_νe, j, νe, j, ρ, M, Fi, Fo, P)
            @views θFD_ρIP_f_nu_nu!(Astar[Jes, Ie + Mc], dcgtt, νe, j, νe, j, ρ, M, Fi, Fo, P, fft_op = :derivative)
            if m != 0
                @views θFD_ρIP_f_nu_nu!(Astar[Jes, Ie + Ms], sgrr, D_νe, j, D_νe, j, ρ, M, Fi, Fo, P)
                @views θFD_ρIP_f_nu_nu!(Astar[Jes, Ie + Ms], sgrt, νe, j, D_νe, j, ρ, M, Fi, Fo, P, fft_op=:derivative)
                @views θFD_ρIP_f_nu_nu!(Astar[Jes, Ie + Ms], dsgrt, D_νe, j, νe, j, ρ, M, Fi, Fo, P)
                @views θFD_ρIP_f_nu_nu!(Astar[Jes, Ie + Ms], dsgtt, νe, j, νe, j, ρ, M, Fi, Fo, P, fft_op = :derivative)
            end

            # Boundary term [νj gρt νi]_0^1, non-zero for νj even and νi even
            # sign flipped to account for later reversal
            if j == 1
                # ρ=0
                @views fourier_decompose!(Astar[Jes, Ie + Mc], θ ->  dcgrt(0.0, θ), M, Fi, Fo, P)
                if m != 0
                    @views fourier_decompose!(Astar[Jes, Ie + Ms], θ ->  dsgrt(0.0, θ), M, Fi, Fo, P)
                end
            elseif j == N
                # ρ=1
                @views fourier_decompose!(Astar[Jes, Ie + Mc], θ -> -dcgrt(1.0, θ), M, Fi, Fo, P)
                if m != 0
                    @views fourier_decompose!(Astar[Jes, Ie + Ms], θ -> -dsgrt(1.0, θ), M, Fi, Fo, P)
                end
            end

            # T[jo, jo+1]
            Io = b2e(shot, jo+1)
            @views θFD_ρIP_f_nu_nu!(Astar[Jos, Io + Mc], cgrr, D_νo, j, D_νe, j, ρ, M, Fi, Fo, P)
            @views θFD_ρIP_f_nu_nu!(Astar[Jos, Io + Mc], cgrt, νo, j, D_νe, j, ρ, M, Fi, Fo, P, fft_op=:derivative)
            @views θFD_ρIP_f_nu_nu!(Astar[Jos, Io + Mc], dcgrt, D_νo, j, νe, j, ρ, M, Fi, Fo, P)
            @views θFD_ρIP_f_nu_nu!(Astar[Jos, Io + Mc], dcgtt, νo, j, νe, j, ρ, M, Fi, Fo, P, fft_op = :derivative)
            if m != 0
                @views θFD_ρIP_f_nu_nu!(Astar[Jos, Io + Ms], sgrr, D_νo, j, D_νe, j, ρ, M, Fi, Fo, P)
                @views θFD_ρIP_f_nu_nu!(Astar[Jos, Io + Ms], sgrt, νo, j, D_νe, j, ρ, M, Fi, Fo, P, fft_op=:derivative)
                @views θFD_ρIP_f_nu_nu!(Astar[Jos, Io + Ms], dsgrt, D_νo, j, νe, j, ρ, M, Fi, Fo, P)
                @views θFD_ρIP_f_nu_nu!(Astar[Jos, Io + Ms], dsgtt, νo, j, νe, j, ρ, M, Fi, Fo, P, fft_op = :derivative)
            end

            if j < length(ρ)
                # T[je, je+1]
                Ie = b2e(shot, je+1)
                @views θFD_ρIP_f_nu_nu!(Astar[Jes, Ie + Mc], cgrr, D_νe, j, D_νo, j+1, ρ, M, Fi, Fo, P)
                @views θFD_ρIP_f_nu_nu!(Astar[Jes, Ie + Mc], cgrt, νe, j, D_νo, j+1, ρ, M, Fi, Fo, P, fft_op=:derivative)
                @views θFD_ρIP_f_nu_nu!(Astar[Jes, Ie + Mc], dcgrt, D_νe, j, νo, j+1, ρ, M, Fi, Fo, P)
                @views θFD_ρIP_f_nu_nu!(Astar[Jes, Ie + Mc], dcgtt, νe, j, νo, j+1, ρ, M, Fi, Fo, P, fft_op = :derivative)
                if m != 0
                    @views θFD_ρIP_f_nu_nu!(Astar[Jes, Ie + Ms], sgrr, D_νe, j, D_νo, j+1, ρ, M, Fi, Fo, P)
                    @views θFD_ρIP_f_nu_nu!(Astar[Jes, Ie + Ms], sgrt, νe, j, D_νo, j+1, ρ, M, Fi, Fo, P, fft_op=:derivative)
                    @views θFD_ρIP_f_nu_nu!(Astar[Jes, Ie + Ms], dsgrt, D_νe, j, νo, j+1, ρ, M, Fi, Fo, P)
                    @views θFD_ρIP_f_nu_nu!(Astar[Jes, Ie + Ms], dsgtt, νe, j, νo, j+1, ρ, M, Fi, Fo, P, fft_op = :derivative)
                end

                # T[jo, jo+2]
                Io = b2e(shot, jo+2)
                @views θFD_ρIP_f_nu_nu!(Astar[Jos, Io + Mc], cgrr, D_νo, j, D_νo, j+1, ρ, M, Fi, Fo, P)
                @views θFD_ρIP_f_nu_nu!(Astar[Jos, Io + Mc], cgrt, νo, j, D_νo, j+1, ρ, M, Fi, Fo, P, fft_op=:derivative)
                @views θFD_ρIP_f_nu_nu!(Astar[Jos, Io + Mc], dcgrt, D_νo, j, νo, j+1, ρ, M, Fi, Fo, P)
                @views θFD_ρIP_f_nu_nu!(Astar[Jos, Io + Mc], dcgtt, νo, j, νo, j+1, ρ, M, Fi, Fo, P, fft_op = :derivative)
                if m != 0
                    @views θFD_ρIP_f_nu_nu!(Astar[Jos, Io + Ms], sgrr, D_νo, j, D_νo, j+1, ρ, M, Fi, Fo, P)
                    @views θFD_ρIP_f_nu_nu!(Astar[Jos, Io + Ms], sgrt, νo, j, D_νo, j+1, ρ, M, Fi, Fo, P, fft_op=:derivative)
                    @views θFD_ρIP_f_nu_nu!(Astar[Jos, Io + Ms], dsgrt, D_νo, j, νo, j+1, ρ, M, Fi, Fo, P)
                    @views θFD_ρIP_f_nu_nu!(Astar[Jos, Io + Ms], dsgtt, νo, j, νo, j+1, ρ, M, Fi, Fo, P, fft_op = :derivative)
                end

                # T[je, je+2]
                Ie = b2e(shot, je+2)
                @views θFD_ρIP_f_nu_nu!(Astar[Jes, Ie + Mc], cgrr, D_νe, j, D_νe, j+1, ρ, M, Fi, Fo, P)
                @views θFD_ρIP_f_nu_nu!(Astar[Jes, Ie + Mc], cgrt, νe, j, D_νe, j+1, ρ, M, Fi, Fo, P, fft_op=:derivative)
                @views θFD_ρIP_f_nu_nu!(Astar[Jes, Ie + Mc], dcgrt, D_νe, j, νe, j+1, ρ, M, Fi, Fo, P)
                @views θFD_ρIP_f_nu_nu!(Astar[Jes, Ie + Mc], dcgtt, νe, j, νe, j+1, ρ, M, Fi, Fo, P, fft_op = :derivative)
                if m != 0
                    @views θFD_ρIP_f_nu_nu!(Astar[Jes, Ie + Ms], sgrr, D_νe, j, D_νe, j+1, ρ, M, Fi, Fo, P)
                    @views θFD_ρIP_f_nu_nu!(Astar[Jes, Ie + Ms], sgrt, νe, j, D_νe, j+1, ρ, M, Fi, Fo, P, fft_op=:derivative)
                    @views θFD_ρIP_f_nu_nu!(Astar[Jes, Ie + Ms], dsgrt, D_νe, j, νe, j+1, ρ, M, Fi, Fo, P)
                    @views θFD_ρIP_f_nu_nu!(Astar[Jes, Ie + Ms], dsgtt, νe, j, νe, j+1, ρ, M, Fi, Fo, P, fft_op = :derivative)
                end

                # T[jo, jo+3]
                Io = b2e(shot, jo+3)
                @views θFD_ρIP_f_nu_nu!(Astar[Jos, Io + Mc], cgrr, D_νo, j, D_νe, j+1, ρ, M, Fi, Fo, P)
                @views θFD_ρIP_f_nu_nu!(Astar[Jos, Io + Mc], cgrt, νo, j, D_νe, j+1, ρ, M, Fi, Fo, P, fft_op=:derivative)
                @views θFD_ρIP_f_nu_nu!(Astar[Jos, Io + Mc], dcgrt, D_νo, j, νe, j+1, ρ, M, Fi, Fo, P)
                @views θFD_ρIP_f_nu_nu!(Astar[Jos, Io + Mc], dcgtt, νo, j, νe, j+1, ρ, M, Fi, Fo, P, fft_op = :derivative)
                if m != 0
                    @views θFD_ρIP_f_nu_nu!(Astar[Jos, Io + Ms], sgrr, D_νo, j, D_νe, j+1, ρ, M, Fi, Fo, P)
                    @views θFD_ρIP_f_nu_nu!(Astar[Jos, Io + Ms], sgrt, νo, j, D_νe, j+1, ρ, M, Fi, Fo, P, fft_op=:derivative)
                    @views θFD_ρIP_f_nu_nu!(Astar[Jos, Io + Ms], dsgrt, D_νo, j, νe, j+1, ρ, M, Fi, Fo, P)
                    @views θFD_ρIP_f_nu_nu!(Astar[Jos, Io + Ms], dsgtt, νo, j, νe, j+1, ρ, M, Fi, Fo, P, fft_op = :derivative)
                end

                # T[je, je+3] does not exist
            end

        end
    end
    # flip sign since all above terms were integrated by parts
    #Astar.nzval .*= -1.0
    return
end

function define_B(shot)
    L = 2 * shot.N * (2 * shot.M + 1)
    B = zeros(L)
    define_B!(B, shot)
    return B
end

function define_B!(B, shot)

    Fi, Fo, P = fft_prealloc(shot.M)
    define_B!(B, shot, Fi, Fo, P)
    return
end

function RHS(shot::Shot, ρ::Real, θ::Real, dp_dψ, f_df_dψ)
    pprime = dp_dψ(ρ, θ)
    ffprim = f_df_dψ(ρ, θ)
    return RHS(shot, ρ, θ, pprime, ffprim)
end

function RHS(shot::Shot, ρ::Real, θ::Real, dp_dψ::Real, f_df_dψ::Real)
    k, nu_ou, nu_eu, nu_ol, nu_el = compute_bases(shot.ρ, ρ)
    R0x = evaluate_inbounds(shot.R0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ϵx = evaluate_inbounds(shot.ϵfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    κx = evaluate_inbounds(shot.κfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    c0x = evaluate_inbounds(shot.c0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    evaluate_csx!(shot, k, nu_ou, nu_eu, nu_ol, nu_el)
    ax = R0x * ϵx

    k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el = compute_D_bases(shot.ρ, ρ)
    dR0x = evaluate_inbounds(shot.R0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dZ0x = evaluate_inbounds(shot.Z0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dϵx = evaluate_inbounds(shot.ϵfe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dκx = evaluate_inbounds(shot.κfe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dc0x = evaluate_inbounds(shot.c0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    evaluate_dcsx!(shot, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    J = MillerExtendedHarmonic.Jacobian(θ, R0x, ϵx, κx, c0x, shot._cx, shot._sx, dR0x, dZ0x, dϵx, dκx, dc0x, shot._dcx, shot._dsx)
    R = MillerExtendedHarmonic.R_MXH(θ, R0x, c0x, shot._cx, shot._sx, ax)

    pterm  =  μ₀ * dp_dψ * J
    ffterm = f_df_dψ * J / R^2

    return -twopi^2 * (pterm + ffterm)
end

function define_B!(B, shot, Fi, Fo, P)
    N = shot.N
    M = shot.M
    ρ = shot.ρ

    B .= 0.0

    mrange = 0:2M
    dp_dψ   = -4.5e4 #-0.1 / μ₀
    f_df_dψ = -3e-2  #-shot.R0fe(1.0)^2

    rhs(x, t) = RHS(shot, x, t, dp_dψ, f_df_dψ)

    # Loop over columns of
    for j in 1:N
        je = 2j
        jo = je - 1
        # element row
        Jo = b2e(shot, jo)
        Je = b2e(shot, je)
        Jos = Jo .+ mrange
        Jes = Je .+ mrange

        @views θFD_ρIP_f_nu!(B[Jos], rhs, νo, j, ρ, M, Fi, Fo, P)
        @views θFD_ρIP_f_nu!(B[Jes], rhs, νe, j, ρ, M, Fi, Fo, P)

    end
    return
end

function set_bc!(shot::Shot, Astar::AbstractMatrix{<:Real}, b::AbstractVector{<:Real})

    N = shot.N
    M = shot.M

    mrange = 0:2M

    # # derivative on-axis is zero
    # Js = 1 .+ mrange
    # Astar[Js, :] .= 0.0
    # for m in Js
    #     Astar[m, m] = 1.0
    # end
    # b[Js] .= 0.0

    # value on edge is zero
    Js = b2e(shot, 2N) .+ mrange
    Astar[Js, :] .= 0.0
    for m in Js
        Astar[m, m] = 1.0
    end
    b[Js] .= 0.0

    # for i in 1:2N
    #     Js = b2e(shot,i)
    #     Astar[Js,:]   .= 0.0
    #     Astar[Js, Js] = 1.0
    #     b[Js] = 0.0
    # end

    return
end