mutable struct Shot{T<:Integer, S<:AbstractVector{<:Real}, R<:AbstractMatrix{<:Real}, Q<:AbstractMatrix{<:Real}} # <: AbstractEquilibrium (eventually)
    N :: T
    M :: T
    ρ :: S
    surfaces :: R
    C :: Q
end

function compute_Cmatrix(N :: Integer, M :: Integer, ρ :: AbstractVector{<:Real}, Ψ_ρθ; Afac::Union{Nothing, AbstractMatrix{<:Real}}=nothing)
    (Afac === nothing) && (Afac = factorize(mass_matrix(N, ρ)))
    C = zeros(2*N,2(M+1))
    for j in 1:N
        C[2j-1,:] = θFD_ρIP_f_nu(Ψ_ρθ, νo, j, ρ, M)
        C[2j  ,:] = θFD_ρIP_f_nu(Ψ_ρθ, νe, j, ρ, M)
    end
    ldiv!(Afac, C)
    return C
end

function Shot(N :: Integer, M :: Integer, ρ :: AbstractVector{<:Real}, surfaces :: AbstractVector{<:MXH}, Ψ;
              zero_boundary=false)
    @assert length(surfaces) == N
    L = length(surfaces[1].c)
    S = zeros(5+2L, N)
    for (k, mxh) in enumerate(surfaces)
       @views flat_coeffs!(S[:, k], mxh)
    end
    Shot(N, M, ρ, S, Ψ; zero_boundary)
end

function Shot(N :: Integer, M :: Integer, ρ :: AbstractVector{<:Real}, surfaces :: AbstractMatrix{<:Real}, Ψ;
    zero_boundary=false)

    S_FE = surfaces_FE(ρ, surfaces)

    function Ψ_ρθ(x, t)
        (zero_boundary && (x == 1.0)) && return 0.0
        return Ψ(R_Z(S_FE..., x, t)...)
    end

    C = compute_Cmatrix(N, M, ρ, Ψ_ρθ)

    return Shot(N, M, ρ, surfaces, C)
end

function Shot(N :: Integer, M :: Integer, boundary :: MXH, Ψ; zero_boundary=false)

    ρ = range(0, 1, N)

    L = length(boundary.c)
    surfaces = zeros(5 + 2L, N)
    for k in eachindex(ρ)
        @views flat_coeffs!(surfaces[:, k], concentric_surface(ρ[k], boundary))
    end

    function Ψ_ρθ(ρ, θ)
        (zero_boundary && ρ == 1.0) && return 0.0
        S = concentric_surface(ρ, boundary)
        return Ψ(S(θ)...)
    end

    C = compute_Cmatrix(N, M, ρ, Ψ_ρθ)

    return Shot(N, M, ρ, surfaces, C)
end

function psi_ρθ(shot::Shot, ρ, θ)
    psi = 0.0
    for i in 1:shot.N
        nuo = νo(ρ,i,shot.ρ)
        nue = νe(ρ,i,shot.ρ)
        for m in 0:shot.M
            sm, cm = sincos(m*θ)
            psi += ((shot.C[2i-1,2m+1] * nuo + shot.C[2i  ,2m+1] * nue) * cm +
                    (shot.C[2i-1,2m+2] * nuo + shot.C[2i  ,2m+2] * nue) * sm)
        end
    end
    return psi
end

function plot_shot(shot::Shot, axes=:rz; points=101)

    p = plot()
    if axes == :ρθ
        xs = range(0, 1, points)
        ys = range(0, 2π, points)
        G = [psi_ρθ(shot, x, y) for y in ys, x in xs]
       heatmap!(p, ys, xs, G')
       contour!(p, ys, xs, G', c=:white)
       plot!(p, xlabel="θ", ylabel="ρ")
    elseif axes == :rz

        @views bnd = shot.surfaces[:, end]
        R0 = bnd[1]
        Z0 = bnd[2]
        ϵ = bnd[3]
        κ = bnd[4]
        a = R0 * ϵ
        Rmin = R0 - a
        Rmax = R0 + a

        Zmin = Z0 - a * κ
        Zmax = Z0 + a * κ
        
        xs = range(Rmin, Rmax, points)
        ys = range(Zmin, Zmax, points)

        S_FE = surfaces_FE(shot)

        G = zeros(points, points)
        for (i,x) in enumerate(xs)
            for (j,y) in enumerate(ys)
                r, z = ρ_θ(S_FE..., x, y)
                if r == NaN
                    G[j,i] = NaN 
                else
                    G[j,i] = psi_ρθ(shot, r, z)
                end
            end
        end
        heatmap!(p, xs, ys, G, aspect_ratio=:equal, clim=(0,1))
    end
    return p
end