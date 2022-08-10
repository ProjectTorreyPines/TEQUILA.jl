mutable struct Shot{T<:Integer, S<:AbstractVector{<:Real}, R<:AbstractVector{<:MXH}, Q<:AbstractMatrix{<:Real}} # <: AbstractEquilibrium (eventually)
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

    surfaces = concentric_surface.(ρ, Ref(boundary))

    function Ψ_ρθ(ρ, θ)
        (zero_boundary && ρ == 1.0) && return 0.0
        S = concentric_surface(ρ, boundary)
        return Ψ(S(θ)...)
    end

    C = compute_Cmatrix(N, M, ρ, Ψ_ρθ)

    return Shot(N, M, ρ, surfaces, C)
end

function psi_ρθ(shot::Shot, ρ, θ)
    return sum(shot.C[2i-1,2m+1] * νo(ρ,i,shot.ρ) * cos(m*θ) +
               shot.C[2i  ,2m+1] * νe(ρ,i,shot.ρ) * cos(m*θ) +
               shot.C[2i-1,2m+2] * νo(ρ,i,shot.ρ) * sin(m*θ) +
               shot.C[2i  ,2m+2] * νe(ρ,i,shot.ρ) * sin(m*θ)
               for i in 1:shot.N, m in 0:shot.M)
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

        boundary = shot.surfaces[end]
        a = boundary.R0 * boundary.ϵ
        Rmin = boundary.R0 - a
        Rmax = boundary.R0 + a

        Zmin = boundary.Z0 - a * boundary.κ
        Zmax = boundary.Z0 + a * boundary.κ
        
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
    elseif :sunflower
        N = points^2
        ys = 2π * 0:(N-1) / MathConstants.golden^2
        xs = sqrt.(range(0,1,N))
        G = g.(zip(xs, ys)...)
        p = nothing
    end
    return p
end

#function (S::Shot)(R, Z)



