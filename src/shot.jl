const ProfType = Union{Nothing, FE_rep, Function}
const IpType = Union{Nothing, Real}

mutable struct Shot{I1<:Integer, VR1<:AbstractVector{<:Real}, MR1<:AbstractMatrix{<:Real}, MR2<:AbstractMatrix{<:Real},
                    PT1<:ProfType, PT2<:ProfType, PT3<:ProfType, PT4<:ProfType, PT5<:ProfType,
                    R1<:Real, R2<:Real, IP1<:IpType,
                    FE1<:FE_rep, VFE1<:AbstractVector{<:FE_rep}, VDC1<:Vector{<:DiffCache}, F1<:Factorization}  <: AbstractEquilibrium
    N :: I1
    M :: I1
    ρ :: VR1
    surfaces :: MR1
    C :: MR2
    P :: PT1
    dP_dψ :: PT2
    F_dF_dψ :: PT3
    Jt_R :: PT4
    Jt :: PT5
    Pbnd :: R1
    Fbnd :: R2
    Ip_target :: IP1
    R0fe::FE1
    Z0fe::FE1
    ϵfe::FE1
    κfe::FE1
    c0fe::FE1
    cfe :: VFE1
    sfe :: VFE1
    _cx :: VDC1
    _sx :: VDC1
    _dcx :: VDC1
    _dsx :: VDC1
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
    @views C[end,:] .= 0.0 # Ensures psi=0 on boundary
    return C
end

function Shot(N :: Integer, M :: Integer, ρ :: AbstractVector{<:Real}, surfaces :: AbstractVector{<:MXH}, Ψ;
              P::ProfType=nothing, dP_dψ::ProfType=nothing,
              F_dF_dψ::ProfType=nothing, Jt_R::ProfType=nothing, Jt::ProfType=nothing,
              Pbnd::Real=0.0, Fbnd::Real=10.0, Ip_target::IpType=nothing, zero_boundary=false)
    @assert length(surfaces) == N
    L = length(surfaces[1].c)
    S = zeros(5+2L, N)
    for (k, mxh) in enumerate(surfaces)
       @views flat_coeffs!(S[:, k], mxh)
    end
    Shot(N, M, ρ, S, Ψ; P, dP_dψ, F_dF_dψ, Jt_R, Jt, Pbnd, Fbnd, Ip_target, zero_boundary)
end

function Shot(N :: Integer, M :: Integer, ρ :: AbstractVector{<:Real}, surfaces :: AbstractMatrix{<:Real}, Ψ;
              P::ProfType=nothing, dP_dψ::ProfType=nothing,
              F_dF_dψ::ProfType=nothing, Jt_R::ProfType=nothing, Jt::ProfType=nothing,
              Pbnd::Real=0.0, Fbnd::Real=10.0, Ip_target::IpType=nothing, zero_boundary=false)

    S_FE = surfaces_FE(ρ, surfaces)

    function Ψ_ρθ(x, t)
        (zero_boundary && (x == 1.0)) && return 0.0
        return Ψ(R_Z(S_FE..., x, t)...)
    end

    C = compute_Cmatrix(N, M, ρ, Ψ_ρθ)

    return Shot(N, M, ρ, surfaces, C, S_FE...; P, dP_dψ, F_dF_dψ, Jt_R, Jt, Pbnd, Fbnd, Ip_target)
end

function Shot(N :: Integer, M :: Integer, boundary :: MXH, Ψ;
              P::ProfType=nothing, dP_dψ::ProfType=nothing,
              F_dF_dψ::ProfType=nothing, Jt_R::ProfType=nothing, Jt::ProfType=nothing,
              Pbnd::Real=0.0, Fbnd::Real=10.0, Ip_target::IpType=nothing, zero_boundary=false)

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

    return Shot(N, M, ρ, surfaces, C, S_FE...; P, dP_dψ, F_dF_dψ, Jt_R, Jt, Pbnd, Fbnd, Ip_target)
end

function Shot(N :: Integer, M :: Integer, boundary :: MXH;
             P::ProfType=nothing, dP_dψ::ProfType=nothing,
             F_dF_dψ::ProfType=nothing, Jt_R::ProfType=nothing, Jt::ProfType=nothing,
             Pbnd::Real=0.0, Fbnd::Real=10.0, Ip_target::IpType=nothing)

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

    return Shot(N, M, ρ, surfaces, C, S_FE...; P, dP_dψ, F_dF_dψ, Jt_R, Jt, Pbnd, Fbnd, Ip_target)
end

function Shot(N :: Integer, M :: Integer, boundary :: MXH, Ψ::FE_rep;
             P::ProfType=nothing, dP_dψ::ProfType=nothing,
             F_dF_dψ::ProfType=nothing, Jt_R::ProfType=nothing, Jt::ProfType=nothing,
             Pbnd::Real=0.0, Fbnd::Real=10.0, Ip_target::IpType=nothing)

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
    C[2:2:end, 1] .= Ψ.(ρ)
    C[1:2:end, 1] .= D.(Ref(Ψ), ρ)

    return Shot(N, M, ρ, surfaces, C, S_FE...; P, dP_dψ, F_dF_dψ, Jt_R, Jt, Pbnd, Fbnd, Ip_target)
end

function Shot(N::Integer, M::Integer, ρ::AbstractVector{<:Real}, surfaces::AbstractMatrix{<:Real},
              C::AbstractMatrix{<:Real}, R0fe::FE_rep, Z0fe::FE_rep, ϵfe::FE_rep, κfe::FE_rep, c0fe::FE_rep,
              cfe::AbstractVector{<:FE_rep}, sfe::AbstractVector{<:FE_rep};
              P::ProfType=nothing, dP_dψ::ProfType=nothing,
              F_dF_dψ::ProfType=nothing, Jt_R::ProfType=nothing, Jt::ProfType=nothing,
              Pbnd::Real=0.0, Fbnd::Real=10.0, Ip_target::IpType=nothing)
    Afac = factorize(mass_matrix(N, ρ))
    return Shot(N, M, ρ, surfaces, C, R0fe, Z0fe, ϵfe, κfe, c0fe, cfe, sfe, Afac;
                P, dP_dψ, F_dF_dψ, Jt_R, Jt, Pbnd, Fbnd, Ip_target)
end

function Shot(N::Integer, M::Integer, ρ::AbstractVector{<:Real}, surfaces::AbstractMatrix{<:Real},
              C::AbstractMatrix{<:Real}, R0fe::FE_rep, Z0fe::FE_rep, ϵfe::FE_rep, κfe::FE_rep, c0fe::FE_rep,
              cfe::AbstractVector{<:FE_rep}, sfe::AbstractVector{<:FE_rep}, Afac::Factorization;
              P::ProfType=nothing, dP_dψ::ProfType=nothing,
              F_dF_dψ::ProfType=nothing, Jt_R::ProfType=nothing, Jt::ProfType=nothing,
              Pbnd::Real=0.0, Fbnd::Real=10.0, Ip_target::IpType=nothing)
    L = length(cfe)
    cx  = [DiffCache(zeros(L)) for _ in 1:Threads.nthreads()]
    sx  = [DiffCache(zeros(L)) for _ in 1:Threads.nthreads()]
    dcx = [DiffCache(zeros(L)) for _ in 1:Threads.nthreads()]
    dsx = [DiffCache(zeros(L)) for _ in 1:Threads.nthreads()]
    Shot(N, M, ρ, surfaces, C, P, dP_dψ, F_dF_dψ, Jt_R, Jt, Pbnd, Fbnd, Ip_target,
         R0fe, Z0fe, ϵfe, κfe, c0fe, cfe, sfe, cx, sx, dcx, dsx, Afac)
end

function Shot(N :: Integer, M :: Integer, MXH_modes::Integer, filename::String; fix_Ip::Bool=false)

    g = MXHEquilibrium.readg(filename)
    cc_in = identify_cocos(g, clockwise_phi=false)[1]
    g = transform_cocos(g, cc_in, 11)
    Ψ = efit(g, 11)

    # boundary
    Rbnd = g.rbbbs
    Zbnd = g.zbbbs
    bnd = MXH(Rbnd, Zbnd, MXH_modes; optimize_fit=true)
    Pbnd = g.pres[end]
    Fbnd = g.fpol[end]

    # profiles
    rho = sqrt.((g.psi .- g.simag) ./ (g.sibry - g.simag))
    dP_dψ = FE(rho, g.pprime)
    F_dF_dψ = FE(rho, g.ffprim)

    Ip_target = fix_Ip ? g.current : nothing

    # Fill a Shot with surfaces concentric to bnd
    return Shot(N, M, bnd, Ψ; dP_dψ, F_dF_dψ, Pbnd, Fbnd, Ip_target)
end

function Shot(shot; P::ProfType=nothing, dP_dψ::ProfType=nothing,
              F_dF_dψ::ProfType=nothing, Jt_R::ProfType=nothing, Jt::ProfType=nothing,
              Pbnd::Real=shot.Pbnd, Fbnd::Real=shot.Fbnd, Ip_target::IpType=shot.Ip_target)
    Np = (P !== nothing) + (dP_dψ !== nothing)
    (Np > 1) && throw(ErrorException("Must specify only one of the following: P, dP_dψ"))
    if Np == 0
        P = deepcopy(shot.P)
        dP_dψ = deepcopy(shot.dP_dψ)
    end

    Nj = (F_dF_dψ !== nothing) + (Jt_R !== nothing) + (Jt !== nothing)
    (Nj > 1) && throw(ErrorException("Must specify only one of the following: F_dF_dψ, Jt_R, Jt"))
    if Nj == 0
        F_dF_dψ = deepcopy(shot.F_dF_dψ)
        Jt_R = deepcopy(shot.Jt_R)
        Jt = deepcopy(shot.Jt)
    end

    return Shot(shot.N, shot.M, deepcopy(shot.ρ), deepcopy(shot.surfaces), deepcopy(shot.C),
                deepcopy(shot.R0fe), deepcopy(shot.Z0fe), deepcopy(shot.ϵfe), deepcopy(shot.κfe),
                deepcopy(shot.c0fe), deepcopy(shot.cfe), deepcopy(shot.sfe);
                P, dP_dψ, F_dF_dψ, Jt_R, Jt, Pbnd, Fbnd, Ip_target)
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

ψ₀(shot::Shot) = shot.C[2,1]
dψ_dρ(shot::Shot, ρ) = -2.0 * ρ * ψ₀(shot)

function ρ(shot::Shot, psi)
    psin = 1.0 - psi / ψ₀(shot)
    return sqrt(psin)
end

function ψ(shot::Shot, ρ)
    psin = ρ ^ 2
    return ψ₀(shot) * (1.0 - psin)
end

@recipe function plot_shot(shot::Shot, axes::Symbol=:rz; points=101, contours=true)

    if axes == :ρθ
        #aspect_ratio --> true
        ρs = range(0, 1, points)
        θs = range(0, twopi, points)
        xguide --> "θ"
        yguide --> "ρ"
        Ψ = [psi_ρθ(shot, ρ, θ) for ρ in ρs, θ in θs]

        @series begin
            seriestype --> :heatmap
            c --> :viridis
            θs, ρs, Ψ
        end
        if contours
            @series begin
                seriestype --> :contour
                c --> :white
                θs, ρs, Ψ
            end
        end
    elseif axes == :rz
        aspect_ratio --> :equal

        bnd = shot.surfaces[:, end]
        Rext, Zext = MXHEquilibrium.limits(shot)
        Rmin, Rmax = Rext
        Zmin, Zmax = Zext

        xlim --> (Rmin, Rmax)
        ylim --> (Zmin, Zmax)

        xs = range(Rmin, Rmax, points)
        ys = range(Zmin, Zmax, points)

        Ψ = zeros(points, points)
        for (i,x) in enumerate(xs)
            for (j,y) in enumerate(ys)
                r, z = ρθ_RZ(shot, x, y)
                if r == NaN
                    Ψ[j,i] = NaN
                else
                    Ψ[j,i] = psi_ρθ(shot, r, z)
                end
            end
        end
        cmap = sum(Ψ) > 0 ? :inferno : cgrad(:inferno, rev=true)
        @series begin
            seriestype --> :heatmap
            c --> cmap
            xs, ys, Ψ
        end
        if contours
            @series begin
                seriestype --> :contour
                colorbar_entry --> false
                linewidth --> 1
                c --> :white
                xs, ys, Ψ
            end
        end
        @series begin
            seriestype --> :path
            linewidth --> 3
            c --> :white
            MXH(bnd)
        end
    end

end

# TEQUILA I/O
save_shot(shot::Shot, filename::String="shot.bson") = BSON.bson(filename, Dict(:shot=>shot))
load_shot(filename::String="shot.bson") = BSON.load(filename)[:shot]

# Implement AbstractEquilibrium interface
# This assume it's converged, so the inner flux surface is the axis
function MXHEquilibrium.magnetic_axis(shot::Shot)
    @views axis = shot.surfaces[:, 1]
    return axis[1], axis[2]
end

function MXHEquilibrium.limits(shot::Shot)
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
    return (Rmin, Rmax), (Zmin, Zmax)
end

function MXHEquilibrium.psi_limits(shot::Shot)
    return ψ₀(shot), 0.0
end

function dpsi_dρ(shot::Shot, ρ, θ, D_bases = compute_D_bases(shot.ρ, ρ))
    dpsi = 0.0
    k = D_bases[1]
    D_nus = D_bases[2:end]

    @views dpsi += dot(shot.C[2k-1:2k+2, 1], D_nus)
    @inbounds for m in 1:shot.M
        @views C = dot(shot.C[2k-1:2k+2, 2m+1], D_nus)
        @views S = dot(shot.C[2k-1:2k+2, 2m], D_nus)
        dpsi += dot((S, C), sincos(m * θ))
    end
    return dpsi
end

function dpsi_dθ(shot::Shot, ρ, θ, bases = compute_bases(shot.ρ, ρ))
    dpsi = 0.0
    k = bases[1]
    nus = bases[2:end]

    # c0 component is zero, no derivative
    @inbounds for m in 1:shot.M
        @views C = dot(shot.C[2k-1:2k+2, 2m+1], nus)
        @views S = dot(shot.C[2k-1:2k+2, 2m], nus)
        dpsi += m * dot((-C, S), sincos(m * θ))
    end
    return dpsi
end

function ∇psi(shot::Shot, ρ, θ, bases_Dbases = compute_both_bases(shot.ρ, ρ))
    bases   = bases_Dbases[1:5]
    D_bases = bases_Dbases[[1,6,7,8,9]]
    return dpsi_dρ(shot, ρ, θ, D_bases), dpsi_dθ(shot, ρ, θ, bases)
end

function MXHEquilibrium.psi_gradient(shot::Shot, R, Z; tid = Threads.threadid())

    ρ, θ = ρθ_RZ(shot, R, Z)

    bases_Dbases = compute_both_bases(shot.ρ, ρ)

    k, nu_ou, nu_eu, nu_ol, nu_el, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el = bases_Dbases

    R0x = evaluate_inbounds(shot.R0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    ϵx = evaluate_inbounds(shot.ϵfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    κx = evaluate_inbounds(shot.κfe, k, nu_ou, nu_eu, nu_ol, nu_el)
    c0x = evaluate_inbounds(shot.c0fe, k, nu_ou, nu_eu, nu_ol, nu_el)
    cx, sx = evaluate_csx!(shot, k, nu_ou, nu_eu, nu_ol, nu_el; tid)

    dR0x = evaluate_inbounds(shot.R0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dZ0x = evaluate_inbounds(shot.Z0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dϵx = evaluate_inbounds(shot.ϵfe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dκx = evaluate_inbounds(shot.κfe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dc0x = evaluate_inbounds(shot.c0fe, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el)
    dcx, dsx = evaluate_dcsx!(shot, k, D_nu_ou, D_nu_eu, D_nu_ol, D_nu_el; tid)

    R_ρ = MillerExtendedHarmonic.dR_dρ(θ, R0x, ϵx, c0x, cx, sx, dR0x, dϵx, dc0x, dcx, dsx)
    R_θ = MillerExtendedHarmonic.dR_dθ(θ, R0x, ϵx, c0x, cx, sx)
    Z_ρ = MillerExtendedHarmonic.dZ_dρ(θ, R0x, ϵx, κx, dR0x, dZ0x, dϵx, dκx)
    Z_θ = MillerExtendedHarmonic.dZ_dθ(θ, R0x, ϵx, κx)
    R_J = R / MillerExtendedHarmonic.Jacobian(θ, R0x, ϵx, κx, c0x, cx, sx, dR0x, dZ0x, dϵx, dκx, dc0x, dcx, dsx)

    Ψ_ρ, Ψ_θ = ∇psi(shot, ρ, θ, bases_Dbases)

    Ψ_R = R_J * (Z_ρ * Ψ_θ - Z_θ * Ψ_ρ)
    Ψ_Z = R_J * (R_θ * Ψ_ρ - R_ρ * Ψ_θ)

    return Ψ_R, Ψ_Z
end

MXHEquilibrium.electric_potential(shot::Shot, psi) = 0.0

MXHEquilibrium.electric_potential_gradient(shot::Shot, psi) = 0.0

function Pprime(shot::Shot, P::Nothing, dP_dψ::Nothing)
    throw(ErrorException("Must specify one of the following: P, dP_dψ"))
end

Pprime(shot::Shot, P::Nothing, dP_dψ) = dP_dψ

function Pprime(shot::Shot, P, dP_dψ::Nothing)
    function dp(x)
        if x == 0.0
            ϵ = 1e-6
            dp1 = dp(ϵ)
            dp2 = dp(2ϵ)
            return 2.0 * dp1 - dp2
        end
        return D(shot.P, x) /  dψ_dρ(shot, x)
    end
    return dp
end

function MXHEquilibrium.pressure_gradient(shot::Shot, psi)
    rho = ρ(shot, psi)
    Pp = Pprime(shot, shot.P, shot.dP_dψ)
    return Pp(rho)
end

function MXHEquilibrium.pressure(shot::Shot, psi)
    rho = ρ(shot, psi)
    if shot.P !== nothing
        return shot.P(rho)
    elseif shot.dP_dψ !== nothing
        f(x) = shot.dP_dψ(x) * dψ_dρ(shot, x)
        return shot.Pbnd - quadgk(f, rho, 1.0)[1]
    end
    throw(ErrorException("Must specify one of the following: P, dP_dψ"))
end


function FFprime(shot::Shot, F_dF_dψ::Nothing, Jt_R::Nothing, Jt::Nothing)
    throw(ErrorException("Must specify one of the following: F_dF_dψ, Jt_R, Jt"))
end

FFprime(shot::Shot, F_dF_dψ, Jt_R::Nothing, Jt::Nothing) = F_dF_dψ

function FFprime(shot::Shot, F_dF_dψ::Nothing, Jt_R, Jt::Nothing)
    invR2 = FE_fsa(shot, fsa_invR2)
    Pp = Pprime(shot, shot.P, shot.dP_dψ)
    return x -> -μ₀ * (Pp(x) + Jt_R(x) / twopi) / invR2(x)
end

function FFprime(shot::Shot, F_dF_dψ::Nothing, Jt_R::Nothing, Jt)
    invR = FE_fsa(shot, fsa_invR)
    invR2 = FE_fsa(shot, fsa_invR2)
    Pp = Pprime(shot, shot.P, shot.dP_dψ)
    return x -> -μ₀ * (Pp(x) + Jt(x) * invR(x) / twopi) / invR2(x)
end

function Fpol_dFpol_dψ(shot::Shot, ρ::Real)
    ffp = FFprime(shot, shot.F_dF_dψ, shot.Jt_R, shot.Jt)
    return ffp(ρ)
end

function Fpol(shot::Shot, ρ::Real)
    return Fpol(shot, FFprime(shot, shot.F_dF_dψ, shot.Jt_R, shot.Jt), ρ)
end

function Fpol(shot::Shot, F_dF_dψ, ρ::Real)
    f(x) = F_dF_dψ(x) * dψ_dρ(shot, x)
    half_dF2 = quadgk(f, ρ, 1.0)[1]
    F2 = shot.Fbnd ^ 2 - 2.0 * half_dF2
    return sign(shot.Fbnd) * sqrt(F2)
end

# Misnomer: "poloidal_current" is actually Fpol = R*Bt, so we'll rename
function MXHEquilibrium.poloidal_current(shot::Shot, psi)
    rho = ρ(shot, psi)
    return Fpol(shot, rho)
end

function dFpol_dψ(shot::Shot, ρ::Real)
    ffp = FFprime(shot, shot.F_dF_dψ, shot.Jt_R, shot.Jt)
    FFp = ffp(ρ)
    Fp = (FFp == 0.0) ? 0.0 : FFp / Fpol(shot, ffp, ρ)
    return Fp
end

# Misnomer: "poloidal_current" is actually Fpol = R*Bt, so we'll rename
function MXHEquilibrium.poloidal_current_gradient(shot::Shot, psi::Real)
    rho = ρ(shot, psi)
    return dFpol_dψ(shot, rho)
end

MXHEquilibrium.cocos(shot::Shot) = MXHEquilibrium.cocos(11)

function MXHEquilibrium.B0Ip_sign(shot::Shot)
    psi = 0.99 * ψ₀(shot)
    signB0 = sign(Fpol(shot, psi))

    sigma_Bp = cocos(shot).sigma_Bp
    signIp = - sign(psi) / sigma_Bp

    return signB0 * signIp

end

MXHEquilibrium.psi_boundary(shot::Shot; kwargs...) = 0.0
