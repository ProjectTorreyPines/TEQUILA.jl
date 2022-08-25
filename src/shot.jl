mutable struct Shot{I1<:Integer, VR1<:AbstractVector{<:Real}, MR1<:AbstractMatrix{<:Real}, MR2<:AbstractMatrix{<:Real},
                    FE1<:FE_rep, VFE1<:AbstractVector{<:FE_rep}} # <: AbstractEquilibrium (eventually)
    N :: I1
    M :: I1
    ρ :: VR1
    surfaces :: MR1
    C :: MR2
    R0fe::FE1
    Z0fe::FE1
    ϵfe::FE1
    κfe::FE1
    c0fe::FE1
    cfe :: VFE1
    sfe :: VFE1
end

function compute_Cmatrix(N :: Integer, M :: Integer, ρ :: AbstractVector{<:Real}, Ψ_ρθ; Afac::Union{Nothing, AbstractMatrix{<:Real}}=nothing)
    (Afac === nothing) && (Afac = factorize(mass_matrix(N, ρ)))
    C = zeros(2*N, 2(M+1))
    for j in 1:N
        @views θFD_ρIP_f_nu!(C[2j-1,:], Ψ_ρθ, νo, j, ρ, M)
        @views θFD_ρIP_f_nu!(C[2j  ,:], Ψ_ρθ, νe, j, ρ, M)
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

    return Shot(N, M, ρ, surfaces, C, S_FE...)
end

function Shot(N :: Integer, M :: Integer, boundary :: MXH, Ψ; zero_boundary=false)

    ρ = range(0, 1, N)

    L = length(boundary.c)
    surfaces = zeros(5 + 2L, N)
    for k in eachindex(ρ)
        @views flat_coeffs!(surfaces[:, k], concentric_surface(ρ[k], boundary))
    end

    S_FE = surfaces_FE(ρ, surfaces)
    tmp_surface = deepcopy(boundary)

    function Ψ_ρθ(ρ, θ)
        (zero_boundary && ρ == 1.0) && return 0.0
        concentric_surface!(tmp_surface, ρ, boundary)
        return Ψ(tmp_surface(θ)...)
    end

    C = compute_Cmatrix(N, M, ρ, Ψ_ρθ)

    return Shot(N, M, ρ, surfaces, C, S_FE...)
end

function psi_ρθ(shot::Shot, ρ, θ)
    psi = 0.0

    k, nu_ou, nu_eu, nu_ol, nu_el = compute_bases(shot.ρ, ρ)
    nus = (nu_ou, nu_eu, nu_ol, nu_el)
    @inbounds for m in 0:shot.M
        @views C = dot(shot.C[2k-1:2k+2, 2m+1], nus)
        @views S = dot(shot.C[2k-1:2k+2, 2m+2], nus)
        psi += dot((S, C), sincos(m * θ))
    end
    return psi
end


function (shot::Shot)(r, z)
    ρ, θ = ρθ_RZ(shot, r, z)
    return psi_ρθ(shot, ρ, θ)
end

function plot_shot(shot::Shot, axes=:rz; points=101)

    p = plot()
    if axes == :ρθ
        xs = range(0, 1, points)
        ys = range(0, twopi, points)
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

function find_axis(shot)
    # First find axis and psi on-axis
    R0 = shot.surfaces[1, 1]
    Z0 = shot.surfaces[2, 1]
    psign = sign(shot(R0, Z0))
    f(x) = -psign * shot(x[1], x[2])
    x0_2[1] = R0
    x0_2[2] = Z0
    S = Optim.optimize(f, x0_2, Optim.NelderMead())
    R0, Z0 = Optim.minimizer(S)
    Ψ0 = -psign * Optim.minimum(S)
    return (R0, Z0, Ψ0)
end