function compute_Cmatrix(N :: Integer, M :: Integer, ρ :: AbstractVector{<:Real}, Ψ::F1, Q::QuadInfo) where {F1}
    return compute_Cmatrix(N, M, ρ, Ψ, Q, factorize(mass_matrix(N, ρ)))
end

function compute_Cmatrix(N :: Integer, M :: Integer, ρ :: AbstractVector{<:Real}, Ψ::F1, Q::QuadInfo, Afac::Factorization) where {F1}
    C = zeros(2N, 2M+1)
    Fi, _, Fo, P = fft_prealloc(M)
    return compute_Cmatrix!(C, N, M, ρ, Ψ, Q, Afac, Fi, Fo, P)
end

function compute_Cmatrix!(C::AbstractMatrix{<:Real}, N :: Integer, M :: Integer, ρ :: AbstractVector{<:Real}, Ψ_ρθ::F1,
                          Q::QuadInfo, Afac::Factorization, Fi::AbstractVector{<:Complex}, Fo::AbstractVector{<:Complex}, P::FFTW.FFTWPlan) where {F1}
    for j in 1:N
        @views θFD_ρIP_f_nu!(C[2j-1, :], Ψ_ρθ, :odd, j, M, Fi, Fo, P, Q)
        @views θFD_ρIP_f_nu!(C[2j  , :], Ψ_ρθ, :even, j, M, Fi, Fo, P, Q)
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

function Ψ_ρθ1(x, t, Ψ, S_FE, zero_boundary)
    (zero_boundary && (x == 1.0)) && return 0.0
    return Ψ(R_Z(S_FE..., x, t)...)
end

function Shot(N :: Integer, M :: Integer, ρ :: AbstractVector{<:Real}, surfaces :: AbstractMatrix{<:Real}, Ψ::F1;
              P::ProfType=nothing, dP_dψ::ProfType=nothing,
              F_dF_dψ::ProfType=nothing, Jt_R::ProfType=nothing, Jt::ProfType=nothing,
              Pbnd::Real=0.0, Fbnd::Real=10.0, Ip_target::IpType=nothing, zero_boundary=false) where {F1}
    S_FE = surfaces_FE(ρ, surfaces)
    MXH_modes = (size(surfaces, 1) - 5) ÷ 2
    Q = QuadInfo(ρ, M, MXH_modes, S_FE...)
    Ψ_ρθ = (x, t) -> Ψ_ρθ1(x, t, Ψ, S_FE, zero_boundary)
    C = compute_Cmatrix(N, M, ρ, Ψ_ρθ, Q)
    return Shot(N, M, ρ, surfaces, C, S_FE..., Q; P, dP_dψ, F_dF_dψ, Jt_R, Jt, Pbnd, Fbnd, Ip_target)
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
    Q = QuadInfo(ρ, M, L, S_FE...)
    tmp_surface = deepcopy(boundary)

    function Ψ_ρθ(ρ, θ)
        (zero_boundary && ρ == 1.0) && return 0.0
        concentric_surface!(tmp_surface, ρ, boundary; Raxis, Zaxis)
        return Ψ(tmp_surface(θ)...)
    end

    C = compute_Cmatrix(N, M, ρ, Ψ_ρθ, Q)

    return Shot(N, M, ρ, surfaces, C, S_FE..., Q; P, dP_dψ, F_dF_dψ, Jt_R, Jt, Pbnd, Fbnd, Ip_target)
end

function Shot(N :: Integer, M :: Integer, boundary :: MXH;
             Raxis::Real = boundary.R0, Zaxis::Real = boundary.Z0,
             P::ProfType=nothing, dP_dψ::ProfType=nothing,
             F_dF_dψ::ProfType=nothing, Jt_R::ProfType=nothing, Jt::ProfType=nothing,
             Pbnd::Real=0.0, Fbnd::Real=10.0, Ip_target::IpType=nothing,
             approximate_psi::Bool=false)

    ρ = range(0, 1, N)

    L = length(boundary.c)
    surfaces = zeros(5 + 2L, N)
    Stmp = deepcopy(boundary)
    for k in eachindex(ρ)
        concentric_surface!(Stmp, ρ[k], boundary; Raxis, Zaxis)
        @views flat_coeffs!(surfaces[:, k], Stmp)
    end

    S_FE = surfaces_FE(ρ, surfaces)
    Q = QuadInfo(ρ, M, L, S_FE...)

    C = zeros(2N, 2M+1)

    shot = Shot(N, M, ρ, surfaces, C, S_FE..., Q; P, dP_dψ, F_dF_dψ, Jt_R, Jt, Pbnd, Fbnd, Ip_target)

    if approximate_psi
        I0 = (Ip_target !== nothing) ? Ip_target : Ip(shot)
        # something like the flux for uniform current density in elliptical wire
        a = boundary.R0 * boundary.ϵ
        ψ0  = 0.5 * μ₀ * boundary.R0 * I0 / sqrt(0.5 * (1.0 + boundary.κ ^ 2))
        C[2:2:end, 1] .= ψ0 .* ((ρ ./ a) .^ 2 .- 1.0)
        C[1:2:end, 1] .= 2.0 .* ψ0 .* ρ ./ (a .^ 2)
        shot = Shot(N, M, ρ, surfaces, C, S_FE..., Q; P, dP_dψ, F_dF_dψ, Jt_R, Jt, Pbnd, Fbnd, Ip_target)
    end

    return shot
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
    Q = QuadInfo(ρ, M, L, S_FE...)

    C = zeros(2N, 2M+1)
    C[2:2:end, 1] .= Ψ.(ρ)
    C[1:2:end, 1] .= D.(Ref(Ψ), ρ)

    return Shot(N, M, ρ, surfaces, C, S_FE..., Q; P, dP_dψ, F_dF_dψ, Jt_R, Jt, Pbnd, Fbnd, Ip_target)
end

function Shot(N::Integer, M::Integer, ρ::AbstractVector{T}, surfaces::AbstractMatrix{<:Real},
              C::AbstractMatrix{<:Real}, R0fe::FE_rep, Z0fe::FE_rep, ϵfe::FE_rep, κfe::FE_rep, c0fe::FE_rep,
              cfe::AbstractVector{<:FE_rep}, sfe::AbstractVector{<:FE_rep}, Q::QuadInfo;
              P::ProfType=nothing, dP_dψ::ProfType=nothing,
              F_dF_dψ::ProfType=nothing, Jt_R::ProfType=nothing, Jt::ProfType=nothing,
              Pbnd::Real=0.0, Fbnd::Real=10.0, Ip_target::IpType=nothing) where {T <: Real}
    Vp    = FE_rep(ρ, Vector{T}(undef, 2N))
    invR  = FE_rep(ρ, Vector{T}(undef, 2N))
    invR2 = FE_rep(ρ, Vector{T}(undef, 2N))
    F  = FE_rep(ρ, Vector{T}(undef, 2N))
    ρtor  = FE_rep(ρ, Vector{T}(undef, 2N))
    # we need an initial ρtor, use ρ
    ρtor.coeffs[1:2:end] .= 1.0
    ρtor.coeffs[2:2:end] .= ρ
    shot =  Shot(N, M, ρ, surfaces, C, R0fe, Z0fe, ϵfe, κfe, c0fe, cfe, sfe, Q, Vp, invR, invR2, F, ρtor;
                 P, dP_dψ, F_dF_dψ, Jt_R, Jt, Pbnd, Fbnd, Ip_target)
    set_FSAs!(shot)
    return shot
end

function Shot(N::Integer, M::Integer, ρ::AbstractVector{<:Real}, surfaces::AbstractMatrix{<:Real},
              C::AbstractMatrix{<:Real}, R0fe::FE_rep, Z0fe::FE_rep, ϵfe::FE_rep, κfe::FE_rep, c0fe::FE_rep,
              cfe::AbstractVector{<:FE_rep}, sfe::AbstractVector{<:FE_rep}, Q::QuadInfo,
              Vp::FE_rep, invR::FE_rep, invR2::FE_rep, F::FE_rep, ρtor::FE_rep;
              P::ProfType=nothing, dP_dψ::ProfType=nothing,
              F_dF_dψ::ProfType=nothing, Jt_R::ProfType=nothing, Jt::ProfType=nothing,
              Pbnd::Real=0.0, Fbnd::Real=10.0, Ip_target::IpType=nothing)
    L = length(cfe)
    cx  = [DiffCache(zeros(L)) for _ in 1:Threads.nthreads()]
    sx  = [DiffCache(zeros(L)) for _ in 1:Threads.nthreads()]
    dcx = [DiffCache(zeros(L)) for _ in 1:Threads.nthreads()]
    dsx = [DiffCache(zeros(L)) for _ in 1:Threads.nthreads()]
    Afac = factorize(mass_matrix(N, ρ))
    MP = prof -> make_profile(prof, ρtor)
    return Shot(N, M, ρ, surfaces, C, MP(P), MP(dP_dψ), MP(F_dF_dψ), MP(Jt_R), MP(Jt), Pbnd, Fbnd, Ip_target,
                R0fe, Z0fe, ϵfe, κfe, c0fe, cfe, sfe, Q, Vp, invR, invR2, F, ρtor, cx, sx, dcx, dsx, Afac)
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
    rho[1] = 0.0
    dP_dψ = (FE(rho, g.pprime), :poloidal)
    F_dF_dψ = (FE(rho, g.ffprim), :poloidal)

    Ip_target = fix_Ip ? g.current : nothing

    # Fill a Shot with surfaces concentric to bnd
    return Shot(N, M, bnd, Ψ; dP_dψ, F_dF_dψ, Pbnd, Fbnd, Ip_target)
end

function Shot(shot; P::ProfType=nothing, dP_dψ::ProfType=nothing,
              F_dF_dψ::ProfType=nothing, Jt_R::ProfType=nothing, Jt::ProfType=nothing,
              Pbnd::Real=shot.Pbnd, Fbnd::Real=shot.Fbnd, Ip_target::IpType=shot.Ip_target)

    Np = (P !== nothing) + (dP_dψ !== nothing)
    if Np == 0
        P = deepcopy(shot.P)
        dP_dψ = deepcopy(shot.dP_dψ)
    elseif Np == 1
        P = make_profile(P, shot.ρtor)
        dP_dψ = make_profile(dP_dψ, shot.ρtor)
    else
        throw(ErrorException("Must specify only one of the following: P, dP_dψ"))
    end

    Nj = (F_dF_dψ !== nothing) + (Jt_R !== nothing) + (Jt !== nothing)
    if Nj == 0
        F_dF_dψ = deepcopy(shot.F_dF_dψ)
        Jt_R = deepcopy(shot.Jt_R)
        Jt = deepcopy(shot.Jt)
    elseif Nj == 1
        F_dF_dψ = make_profile(F_dF_dψ, shot.ρtor)
        Jt_R = make_profile(Jt_R, shot.ρtor)
        Jt = make_profile(Jt, shot.ρtor)
    else
        throw(ErrorException("Must specify only one of the following: F_dF_dψ, Jt_R, Jt"))
    end

    return Shot(shot.N, shot.M, deepcopy(shot.ρ), deepcopy(shot.surfaces), deepcopy(shot.C),
                deepcopy(shot.R0fe), deepcopy(shot.Z0fe), deepcopy(shot.ϵfe), deepcopy(shot.κfe),
                deepcopy(shot.c0fe), deepcopy(shot.cfe), deepcopy(shot.sfe), deepcopy(shot.Q);
                P, dP_dψ, F_dF_dψ, Jt_R, Jt, Pbnd, Fbnd, Ip_target)
end

function psi_ρθ(shot::Shot, ρ::Real, θ::Real)
    psi = 0.0

    k, nu_ou, nu_eu, nu_ol, nu_el = compute_bases(shot.ρ, ρ)
    tk = 2k
    tkm1 = tk-1
    tkp1 = tk+1
    tkp2 = tk+2

    psi += shot.C[tkm1, 1] * nu_ou + shot.C[tk, 1] * nu_eu + shot.C[tkp1, 1] * nu_ol + shot.C[tkp2, 1] * nu_el
    shot.M == 0 && return psi

    m = 1
    tm = 2
    S = shot.C[tkm1, tm] * nu_ou + shot.C[tk, tm] * nu_eu + shot.C[tkp1, tm] * nu_ol + shot.C[tkp2, tm] * nu_el
    C = shot.C[tkm1, tm+1] * nu_ou + shot.C[tk, tm+1] * nu_eu + shot.C[tkp1, tm+1] * nu_ol + shot.C[tkp2, tm+1] * nu_el
    s1, c1 = sincos(θ)
    psi += S * s1 + C * c1
    shot.M == 1 && return psi

    m = 2
    tm = 4
    S = shot.C[tkm1, tm] * nu_ou + shot.C[tk, tm] * nu_eu + shot.C[tkp1, tm] * nu_ol + shot.C[tkp2, tm] * nu_el
    C = shot.C[tkm1, tm+1] * nu_ou + shot.C[tk, tm+1] * nu_eu + shot.C[tkp1, tm+1] * nu_ol + shot.C[tkp2, tm+1] * nu_el
    tc1 = 2.0 * c1
    s2 = tc1 * s1        # sin(2θ)
    c2 = tc1 * c1 - 1.0  # cos(2θ)
    psi += S * s2 + C * c2
    shot.M == 2 && return psi

    # Chebyshev method for recursively computing sin(mθ) and cos(mθ)
    # https://en.wikipedia.org/wiki/List_of_trigonometric_identities#Chebyshev_method
    sm_1, cm_1 = s1, c1
    sm, cm = s2, c2
    @inbounds for m in 3:shot.M
        # The m-th Fourier coefficient at this ρ
        tm = 2m
        S = shot.C[tkm1, tm] * nu_ou + shot.C[tk, tm] * nu_eu + shot.C[tkp1, tm] * nu_ol + shot.C[tkp2, tm] * nu_el
        C = shot.C[tkm1, tm+1] * nu_ou + shot.C[tk, tm+1] * nu_eu + shot.C[tkp1, tm+1] * nu_ol + shot.C[tkp2, tm+1] * nu_el

        sm_2, cm_2 = sm_1, cm_1
        sm_1, cm_1 = sm, cm
        cm = tc1 * cm_1 - cm_2 # cos(mθ)
        sm = tc1 * sm_1 - sm_2 # sin(mθ)
        psi += S * sm + C * cm
    end
    return psi
end

function psi_ρθ(shot::Shot, ρ::Real, Fsin::AbstractVector{<:Real}, Fcos::AbstractVector{<:Real})
    psi = 0.0

    k, nu_ou, nu_eu, nu_ol, nu_el = compute_bases(shot.ρ, ρ)
    nus = (nu_ou, nu_eu, nu_ol, nu_el)

    @views psi += dot(shot.C[2k-1:2k+2, 1], nus)
    @inbounds for m in 1:shot.M
        @views C = dot(shot.C[2k-1:2k+2, 2m+1], nus)
        @views S = dot(shot.C[2k-1:2k+2, 2m], nus)
        psi += S * Fsin[m] + C * Fcos[m]
    end
    return psi
end


function (shot::Shot)(r, z; extrapolate::Bool=false)
    ρ, θ = ρθ_RZ(shot, r, z; extrapolate)
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

@recipe function plot_shot(shot::Shot, axes::Symbol=:rz; points=101, contours=true, surfaces=false, extrapolate=false)

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

        Ψ = [shot(x,  y; extrapolate) for y in ys, x in xs]

        pmin, pmax = extrema(Ψ)


        if pmin < 0 && pmax > 0
            cmap = :diverging
            pext = max(abs(pmin), abs(pmax))
            clims --> (-pext, pext)
        elseif pmax > 0
            cmap = :inferno
        else
            cmap = cgrad(:inferno, rev=true)
        end
        @series begin
            seriestype --> :heatmap
            c --> cmap
            xs, ys, Ψ
        end
        if contours
            @series begin
                seriestype --> :contour
                colorbar_entry --> false
                linewidth --> 2
                c --> :white
                xs, ys, Ψ
            end
        end
        if surfaces
            for flat in eachcol(shot.surfaces)[2:end]
                @series begin
                    seriestype --> :path
                    c --> :cyan
                    linewidth --> 1
                    MXH(flat)
                end
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

function dpsi_dρ(shot::Shot, ρ::Real, θ::Real, D_bases = compute_D_bases(shot.ρ, ρ))
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

function dpsi_dρ(shot::Shot, ρ::Real, Fsin::AbstractVector{<:Real}, Fcos::AbstractVector{<:Real}, D_bases = compute_D_bases(shot.ρ, ρ))
    dpsi = 0.0
    k = D_bases[1]
    D_nus = D_bases[2:end]

    @views dpsi += dot(shot.C[2k-1:2k+2, 1], D_nus)
    @inbounds for m in 1:shot.M
        @views C = dot(shot.C[2k-1:2k+2, 2m+1], D_nus)
        @views S = dot(shot.C[2k-1:2k+2, 2m], D_nus)
        dpsi += S * Fsin[m] + C * Fcos[m]
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
    @views bases   = bases_Dbases[1:5]
    @views D_bases = bases_Dbases[1], bases_Dbases[6:9]...
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

function _dp(x, shot)
    if x == 0.0
        ϵ = 1e-6
        dp1 = _dp(ϵ, shot)
        dp2 = _dp(2ϵ, shot)
        return 2.0 * dp1 - dp2
    end
    ψprime = dψ_dρ(shot, x)
    pprime = (ψprime == 0.0) ? 0.0 : deriv(shot.P, x) /  ψprime
    return pprime
end
Pprime(shot::Shot, P, dP_dψ::Nothing) = (x -> _dp(x, shot))

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

FFprime(shot::Shot, F_dF_dψ, Jt_R::Nothing, Jt::Nothing; invR = nothing, invR2=nothing) = F_dF_dψ

function FFprime(shot::Shot, F_dF_dψ::Nothing, Jt_R, Jt::Nothing; invR = nothing, invR2 = FE_rep(shot, fsa_invR2))
    Pp = Pprime(shot, shot.P, shot.dP_dψ)
    return x -> -μ₀ * (Pp(x) + Jt_R(x) / twopi) / invR2(x)
end

function FFprime(shot::Shot, F_dF_dψ::Nothing, Jt_R::Nothing, Jt; invR = FE_rep(shot, fsa_invR), invR2 = FE_rep(shot, fsa_invR2))
    Pp = Pprime(shot, shot.P, shot.dP_dψ)
    return x -> -μ₀ * (Pp(x) + Jt(x) * invR(x) / twopi) / invR2(x)
end

function Fpol_dFpol_dψ(shot::Shot, ρ::Real; kwargs...)
    ffp = FFprime(shot, shot.F_dF_dψ, shot.Jt_R, shot.Jt; kwargs...)
    return ffp(ρ)
end

function Fpol(shot::Shot, ρ::Real; kwargs...)
    return Fpol(shot, FFprime(shot, shot.F_dF_dψ, shot.Jt_R, shot.Jt; kwargs...), ρ)
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

MXHEquilibrium.plasma_current(shot::Shot) = Ip(shot)