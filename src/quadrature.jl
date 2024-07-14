function QuadInfo(shot::Shot; order::Int=int_order)
    MXH_modes = length(shot.cfe)
    Q = QuadInfo(shot.ρ, shot.M, MXH_modes; order)
    MXH_quadrature!(Q, shot)
    metrics_quadrature!(Q)
    return Q
end

function QuadInfo(ρ::AbstractVector, M::Int, MXH_modes::Int; order::Int=int_order)
    # ρ only
    xq, wq = get_quadrature(ρ, order)
    Nρ = length(xq)
    nuo = [nu_quadrature(νo, xq, k, ρ, order) for k in eachindex(ρ)]
    nue = [nu_quadrature(νe, xq, k, ρ, order) for k in eachindex(ρ)]
    D_nuo = [nu_quadrature(D_νo, xq, k, ρ, order) for k in eachindex(ρ)]
    D_nue = [nu_quadrature(D_νe, xq, k, ρ, order) for k in eachindex(ρ)]

    Nq = length(xq)
    R0 = similar(xq)
    Z0 = similar(xq)
    ϵ = similar(xq)
    κ = similar(xq)
    c0 = similar(xq)
    c = zeros(MXH_modes, Nq)
    s = zeros(MXH_modes, Nq)

    dR0 = similar(xq)
    dZ0 = similar(xq)
    dϵ = similar(xq)
    dκ = similar(xq)
    dc0 = similar(xq)
    dc = zeros(MXH_modes, Nq)
    ds = zeros(MXH_modes, Nq)

    # θ only
    θ = range(0, twopi, 2M + 5)[1:end-1]
    Mθ = 2M + 4
    Mrows = max(Mθ, MXH_modes)
    Fsin = zeros(Mrows, Mθ)
    Fcos = zero(Fsin)
    for l in eachindex(θ)
        for m in 1:Mrows
            Fsin[m, l], Fcos[m, l] = sincos(m * θ[l])
        end
    end

    # ρ and θ
    gρρ = zeros(Nρ, Mθ)
    gρθ = similar(gρρ)
    gθθ = similar(gρρ)

    return QuadInfo(xq, wq, nuo, nue, D_nuo, D_nue, R0, Z0, ϵ, κ, c0, c, s, dR0, dZ0, dϵ, dκ, dc0, dc, ds, θ, Fsin, Fcos, gρρ, gρθ, gθθ)
end

function QuadInfo(ρ::AbstractVector, M::Int, MXH_modes::Int, R0fe, Z0fe, ϵfe, κfe, c0fe, cfe, sfe; order::Int=int_order)
    Q = QuadInfo(ρ, M, MXH_modes; order)
    MXH_quadrature!(Q, ρ, R0fe, Z0fe, ϵfe, κfe, c0fe, cfe, sfe)
    metrics_quadrature!(Q)
    return Q
end

function nu_quadrature(nu::Function, xq::AbstractVector{<:Real}, k::Int, ρ::AbstractVector{<:Real}, order::Int)
    N = length(ρ)
    Nq = length(xq)
    if k == 1
        J = range(1, 5)
    elseif k == N
        J = range(Nq - 4, Nq)
    else
        J = range(-4, 5) .+ order * (k - 1)
    end
    @views V = nu.(xq[J], Ref(k), Ref(ρ))
    return sparsevec(J, V, Nq)
end

function get_nu(Q::QuadInfo, nu::Symbol, k::Int)
    @assert nu in (:odd, :even, :D_odd, :D_even)
    if nu === :odd
        return Q.νo[k]
    elseif nu === :even
        return Q.νe[k]
    elseif nu === :D_odd
        return Q.D_νo[k]
    else
        return Q.D_νe[k]
    end
end

MXH_quadrature!(shot::Shot) = MXH_quadrature!(shot.Q, shot)

function MXH_quadrature!(Q::QuadInfo, shot::Shot)
    for (k, ρ) in enumerate(Q.x)
        R0x, Z0x, ϵx, κx, c0x, cx, sx, dR0x, dZ0x, dϵx, dκx, dc0x, dcx, dsx = compute_both_MXH(shot, ρ)
        Q.R0[k] = R0x
        Q.Z0[k] = Z0x
        Q.ϵ[k] = ϵx
        Q.κ[k] = κx
        Q.c0[k] = c0x
        Q.c[:, k] .= cx
        Q.s[:, k] .= sx
        Q.dR0[k] = dR0x
        Q.dZ0[k] = dZ0x
        Q.dϵ[k] = dϵx
        Q.dκ[k] = dκx
        Q.dc0[k] = dc0x
        Q.dc[:, k] .= dcx
        Q.ds[:, k] .= dsx
    end
    return Q
end

function MXH_quadrature!(Q::QuadInfo, ρs::AbstractVector{<:Real}, R0fe, Z0fe, ϵfe, κfe, c0fe, cfe, sfe)
    for (k, ρ) in enumerate(Q.x)
        R0x, Z0x, ϵx, κx, c0x, cx, sx, dR0x, dZ0x, dϵx, dκx, dc0x, dcx, dsx = compute_both_MXH(ρs, ρ, R0fe, Z0fe, ϵfe, κfe, c0fe, cfe, sfe)
        Q.R0[k] = R0x
        Q.Z0[k] = Z0x
        Q.ϵ[k] = ϵx
        Q.κ[k] = κx
        Q.c0[k] = c0x
        Q.c[:, k] .= cx
        Q.s[:, k] .= sx
        Q.dR0[k] = dR0x
        Q.dZ0[k] = dZ0x
        Q.dϵ[k] = dϵx
        Q.dκ[k] = dκx
        Q.dc0[k] = dc0x
        Q.dc[:, k] .= dcx
        Q.ds[:, k] .= dsx
    end
    return Q
end

get_MXH_quadrature(shot::Shot, k::Int) = get_MXH(shot.Q, k)

function get_MXH(Q::QuadInfo, k::Int)
    R0 = Q.R0[k]
    Z0 = Q.Z0[k]
    ϵ = Q.ϵ[k]
    κ = Q.κ[k]
    c0 = Q.c0[k]
    c = @views Q.c[:, k]
    s = @views Q.s[:, k]
    dR0 = Q.dR0[k]
    dZ0 = Q.dZ0[k]
    dϵ = Q.dϵ[k]
    dκ = Q.dκ[k]
    dc0 = Q.dc0[k]
    dc = @views Q.dc[:, k]
    ds = @views Q.ds[:, k]
    return R0, Z0, ϵ, κ, c0, c, s, dR0, dZ0, dϵ, dκ, dc0, dc, ds
end

metrics_quadrature!(shot::Shot) = metrics_quadrature!(shot.Q)

function metrics_quadrature!(Q::QuadInfo)
    for (k, ρ) in enumerate(Q.x)
        R0x, _, ϵx, κx, c0x, cx, sx, dR0x, dZ0x, dϵx, dκx, dc0x, dcx, dsx = get_MXH(Q, k)
        for (j, θ) in enumerate(Q.θ)
            grr, grt, gtt = MillerExtendedHarmonic.gρρ_gρθ_gθθ(θ, R0x, ϵx, κx, c0x, cx, sx, dR0x, dZ0x, dϵx, dκx, dc0x, dcx, dsx, Q.Fsin, Q.Fcos)
            Q.gρρ[k, j] = grr
            Q.gρθ[k, j] = grt
            Q.gθθ[k, j] = gtt
        end
    end
    return Q
end

get_metrics_quadrature(shot::Shot, j::Int, k::Int) = get_metrics(shot.Q, j, k)

function get_metrics(Q::QuadInfo, j::Int, k::Int)
    return Q.gρρ[k, j], Q.gρθ[k, j], Q.gθθ[k, j]

end