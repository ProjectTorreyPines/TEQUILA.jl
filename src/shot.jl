mutable struct Shot{I1<:Integer, VR1<:AbstractVector{<:Real}, MR1<:AbstractMatrix{<:Real}, MR2<:AbstractMatrix{<:Real},
                    FE1<:FE_rep, VFE1<:AbstractVector{<:FE_rep}, VR2<:AbstractVector{<:Real}, F1<:Factorization} # <: AbstractEquilibrium (eventually)
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
    _cx :: VR2
    _sx :: VR2
    _dcx :: VR2
    _dsx :: VR2
    _Afac :: F1
end

function compute_Cmatrix(N :: Integer, M :: Integer, ρ :: AbstractVector{<:Real}, Ψ::Union{Function, Shot})
    return compute_Cmatrix(N, M, ρ, Ψ, factorize(mass_matrix(N, ρ)))
end

function compute_Cmatrix(N :: Integer, M :: Integer, ρ :: AbstractVector{<:Real}, Ψ::Union{Function, Shot}, Afac::Factorization)
    C = zeros(2N, 2M+1)
    Fi, _, Fo, P = fft_prealloc(M)
    return compute_Cmatrix!(C, N, M, ρ, Ψ, Afac, Fi, Fo, P)
end

function compute_Cmatrix!(C::AbstractMatrix{<:Real}, N :: Integer, M :: Integer, ρ :: AbstractVector{<:Real}, Ψ_ρθ,
                          Afac::Factorization, Fi::AbstractVector{<:Complex}, Fo::AbstractVector{<:Complex}, P::FFTW.FFTWPlan)
    for j in 1:N
        @views θFD_ρIP_f_nu!(C[2j-1, :], Ψ_ρθ, νo, j, ρ, M, Fi, Fo, P)
        @views θFD_ρIP_f_nu!(C[2j  , :], Ψ_ρθ, νe, j, ρ, M, Fi, Fo, P)
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

    Raxis, Zaxis, _ = find_axis(Ψ, boundary.R0, boundary.Z0)

    L = length(boundary.c)
    surfaces = zeros(5 + 2L, N)
    Stmp = deepcopy(boundary)
    for k in eachindex(ρ)
        concentric_surface!(Stmp, ρ[k], boundary; Raxis, Zaxis)
        @views flat_coeffs!(surfaces[:, k], Stmp)
    end

    S_FE = surfaces_FE(ρ, surfaces)
    tmp_surface = deepcopy(boundary)

    function Ψ_ρθ(ρ, θ)
        (zero_boundary && ρ == 1.0) && return 0.0
        concentric_surface!(tmp_surface, ρ, boundary; Raxis, Zaxis)
        return Ψ(tmp_surface(θ)...)
    end

    C = compute_Cmatrix(N, M, ρ, Ψ_ρθ)

    return Shot(N, M, ρ, surfaces, C, S_FE...)
end

function Shot(N :: Integer, M :: Integer, boundary :: MXH)

    ρ = range(0, 1, N)

    L = length(boundary.c)
    surfaces = zeros(5 + 2L, N)
    Stmp = deepcopy(boundary)
    for k in eachindex(ρ)
        concentric_surface!(Stmp, ρ[k], boundary)
        @views flat_coeffs!(surfaces[:, k], Stmp)
    end

    S_FE = surfaces_FE(ρ, surfaces)

    C = zeros(2N, 2M+1)

    return Shot(N, M, ρ, surfaces, C, S_FE...)
end

function Shot(N::Integer, M::Integer, ρ::AbstractVector{<:Real}, surfaces::AbstractMatrix{<:Real},
              C::AbstractMatrix{<:Real}, R0fe::FE_rep, Z0fe::FE_rep, ϵfe::FE_rep, κfe::FE_rep, c0fe::FE_rep,
              cfe::AbstractVector{<:FE_rep}, sfe::AbstractVector{<:FE_rep})
    Afac = factorize(mass_matrix(N, ρ))
    return Shot(N, M, ρ, surfaces, C, R0fe, Z0fe, ϵfe, κfe, c0fe, cfe, sfe, Afac)
end

function Shot(N::Integer, M::Integer, ρ::AbstractVector{<:Real}, surfaces::AbstractMatrix{<:Real},
              C::AbstractMatrix{<:Real}, R0fe::FE_rep, Z0fe::FE_rep, ϵfe::FE_rep, κfe::FE_rep, c0fe::FE_rep,
              cfe::AbstractVector{<:FE_rep}, sfe::AbstractVector{<:FE_rep}, Afac::Factorization)
    L = length(cfe)
    cx = zeros(L)
    sx = zeros(L)
    dcx = zeros(L)
    dsx = zeros(L)
    Shot(N, M, ρ, surfaces, C, R0fe, Z0fe, ϵfe, κfe, c0fe, cfe, sfe, cx, sx, dcx, dsx, Afac)
end

function Shot(N :: Integer, M :: Integer, MXH_modes::Integer, filename::String)

    g = MXHEquilibrium.readg(filename)
    cc_in = identify_cocos(g, clockwise_phi=false)[1]
    g = transform_cocos(g, cc_in, 11)
    Ψ = efit(g, 11)

    # boundary
    Rbnd = g.rbbbs
    Zbnd = g.zbbbs
    bnd = MXH(Rbnd, Zbnd, MXH_modes; optimize_fit=true)

    # Fill a Shot with surfaces concentric to bnd
    return Shot(N, M, bnd, Ψ);
end

# function remap_shot!(shot::Shot, surfaces :: AbstractMatrix{<:Real})
#     Fi, _, Fo, P = fft_prealloc(shot.M)
#     return remap_shot!(shot, surfaces, Fi, Fo, P)
# end

# function remap_shot!(shot::Shot, surfaces :: AbstractMatrix{<:Real}, Fi::AbstractVector{<:Complex}, Fo::AbstractVector{<:Complex}, P::FFTW.FFTWPlan)

#     compute_Cmatrix!(shot.C, shot.N, shot.M, shot.ρ, shot, shot._Afac, Fi, Fo, P)
#     #shot.C .= compute_Cmatrix(N, M, ρ, shot)
#     shot.surfaces .= surfaces
#     shot.R0fe, shot.Z0fe, shot.ϵfe, shot.κfe, shot.c0fe, shot.cfe, shot.sfe = surfaces_FE(ρ, surfaces)
#     return shot
# end

function psi_ρθ(shot::Shot, ρ, θ)
    psi = 0.0

    k, nu_ou, nu_eu, nu_ol, nu_el = compute_bases(shot.ρ, ρ)
    nus = (nu_ou, nu_eu, nu_ol, nu_el)

    @views psi += dot(shot.C[2k-1:2k+2, 1], nus)
    @inbounds for m in 1:shot.M
        @views C = dot(shot.C[2k-1:2k+2, 2m+1], nus)
        @views S = dot(shot.C[2k-1:2k+2, 2m], nus)
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
       heatmap!(p, ys, xs, G', c=:viridis)
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

        G = zeros(points, points)
        for (i,x) in enumerate(xs)
            for (j,y) in enumerate(ys)
                r, z = ρθ_RZ(shot, x, y)
                if r == NaN
                    G[j,i] = NaN
                else
                    G[j,i] = psi_ρθ(shot, r, z)
                end
            end
        end
        heatmap!(p, xs, ys, G, aspect_ratio=:equal)#, clim=(0,1))
    end
    return p
end

function find_axis(shot)
    R0 = shot.surfaces[1, 1]
    Z0 = shot.surfaces[2, 1]
    return find_axis(shot, R0, Z0)
end

function find_axis(Ψ, R0::Real, Z0::Real)
    psign = sign(Ψ(R0, Z0))
    f(x) = -psign * Ψ(x[1], x[2])
    x0_2[1] = R0
    x0_2[2] = Z0
    S = Optim.optimize(f, x0_2, Optim.NelderMead())
    Raxis, Zaxis = Optim.minimizer(S)
    Ψaxis = -psign * Optim.minimum(S)
    return (Raxis, Zaxis, Ψaxis)
end