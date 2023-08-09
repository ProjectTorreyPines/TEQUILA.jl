f_nu_mu(x, θ, f, nu, k, mu, m, ρ) = f(x, θ) * nu(x, k, ρ) * mu(m*θ)

"""
    fourier_decompose(f, M::Integer)

Decompose f(θ) into a Fourier series with m from 0 to M
Returns cosine coeffients and sine coefficients as tuple
"""
function fft_prealloc(M::Integer)
    tmp = zeros(2M+4)
    Fi = complex(tmp)
    dFi = complex(tmp)
    Fo = complex(tmp)
    P = plan_fft(Fi)
    return Fi, dFi, Fo, P
end

function fft_prealloc_threaded(M::Integer)
    tmp = zeros(2M+4)
    Fis  = [complex(tmp) for _ in 1:Threads.nthreads()]
    dFis = [complex(tmp) for _ in 1:Threads.nthreads()]
    Fos  = [complex(tmp) for _ in 1:Threads.nthreads()]
    Ps   = [plan_fft(complex(tmp))  for _ in 1:Threads.nthreads()]
    return Fis, dFis, Fos, Ps
end

function fourier_decompose(f, M::Integer; fft_op::Union{Nothing, Symbol}=nothing)
    CS = zeros(2M + 1)
    Fi, _, Fo, P = fft_prealloc(M)
    return fourier_decompose!(CS, f, M, Fi, Fo, P; fft_op)
end

function fourier_decompose!(CS::AbstractVector{<:Real}, f::F1, M::Integer, Fi::AbstractVector{<:Complex},
                            Fo::AbstractVector{<:Complex}, P::FFTW.FFTWPlan, Q::Nothing=nothing;
                            reset_CS = false, fft_op::Union{Nothing, Symbol}=nothing) where {F1<:Function}
    θs = range(0,twopi, 2M+5)[1:end-1]
    return fourier_decompose!(CS, f, M, Fi, Fo, P, θs; reset_CS, fft_op)
end

function fourier_decompose!(CS::AbstractVector{<:Real}, f::F1, M::Integer, Fi::AbstractVector{<:Complex},
                            Fo::AbstractVector{<:Complex}, P::FFTW.FFTWPlan, Q::QuadInfo;
                            reset_CS = false, fft_op::Union{Nothing, Symbol}=nothing) where {F1<:Function}
    return fourier_decompose!(CS, f, M, Fi, Fo, P, Q.θ; reset_CS, fft_op)
end

function fourier_decompose!(CS::AbstractVector{<:Real}, f::F1, M::Integer, Fi::AbstractVector{<:Complex},
                            Fo::AbstractVector{<:Complex}, P::FFTW.FFTWPlan, θs::AbstractVector{<:Real};
                            reset_CS = false, fft_op::Union{Nothing, Symbol}=nothing) where {F1<:Function}
    invM2 = 1.0 / (M + 2)
    @. Fi = f(θs)
    mul!(Fo, P, Fi)
    reset_CS && (CS .= 0.0)
    if fft_op === :derivative
        m_M2 = range(0, M * invM2, M+1)
        @views CS[1:2:end] .+= m_M2 .* imag.(Fo[1:(M+1)])
        @views CS[2:2:end] .+= m_M2[2:end] .* real.(Fo[2:(M+1)])
    else
        Fo[1] *= 0.5
        @views CS[1:2:end] .+=  real.(Fo[1:(M+1)]) .* invM2
        @views CS[2:2:end] .+= .-imag.(Fo[2:(M+1)]) .* invM2 # fft sign convention
    end
    return CS
end

function dual_fourier_decompose!(CS::AbstractVector{<:Real}, f, M::Integer,
                                 Fi::AbstractVector{<:Complex}, dFi::AbstractVector{<:Complex},
                                 Fo::AbstractVector{<:Complex}, P::FFTW.FFTWPlan, Q::Nothing=nothing;
                                 reset_CS = false)
    θs = range(0, twopi, 2M+5)[1:end-1]
    return dual_fourier_decompose!(CS, f, M, Fi, dFi, Fo, P, θs; reset_CS)
end

function dual_fourier_decompose!(CS::AbstractVector{<:Real}, f, M::Integer,
                                 Fi::AbstractVector{<:Complex}, dFi::AbstractVector{<:Complex},
                                 Fo::AbstractVector{<:Complex}, P::FFTW.FFTWPlan, Q::QuadInfo;
                                 reset_CS = false)
    return dual_fourier_decompose!(CS, f, M, Fi, dFi, Fo, P, Q.θ; reset_CS)
end

function dual_fourier_decompose!(CS::AbstractVector{<:Real}, f, M::Integer,
                                 Fi::AbstractVector{<:Complex}, dFi::AbstractVector{<:Complex},
                                 Fo::AbstractVector{<:Complex}, P::FFTW.FFTWPlan, θs::AbstractVector{<:Real};
                                 reset_CS = false)
    reset_CS && (CS .= 0.0)
    invM2 = 1.0 / (M + 2)

    for (k, θ) in enumerate(θs)
        Fi[k], dFi[k] = f(θ)
    end

    mul!(Fo, P, Fi)
    Fo[1] *= 0.5
    @views CS[1:2:end] .+=  real.(Fo[1:(M+1)]) .* invM2
    @views CS[2:2:end] .+= .-imag.(Fo[2:(M+1)]) .* invM2 # fft sign convention

    mul!(Fo, P, dFi)
    m_M2 = range(0, M * invM2, M+1)
    @views CS[1:2:end] .+= m_M2 .* imag.(Fo[1:(M+1)])
    @views CS[2:2:end] .+= m_M2[2:end] .* real.(Fo[2:(M+1)])

    return CS
end


# At fixed θ, give inner product of f(x,θ) and the basis nu(x,k,ρ)
@inline function ρIP(θ, f::F1, nu::F2, k, ρ, Q::Nothing=nothing) where {F1<:Union{Function, Shot}, F2<:Function}
    return inner_product(x -> f(x, θ), nu, k, ρ, int_order)
end

function ρIP(θ, f, nu::Symbol, k, ρ, Q::QuadInfo)
    ν = get_nu(Q, nu, k)
    return sum(j -> Q.w[j] * f(Q.x[j], θ) * ν[j], rowvals(ν))
end

@inline ρIP(θ, f, nu1, k1, nu2, k2, ρ, Q::Nothing=nothing) = inner_product(x -> f(x, θ), nu1, k1, nu2, k2, ρ, int_order)

function ρIP(θ, f, nu1::Symbol, k1, nu2::Symbol, k2, ρ, Q::QuadInfo)
    abs(k1 - k2) > 1 && return 0.0
    ν1 = get_nu(Q, nu1, k1)
    ν2 = get_nu(Q, nu2, k2)
    jmin = max(minimum(rowvals(ν1)), minimum(rowvals(ν2)))
    jmax = min(maximum(rowvals(ν1)), maximum(rowvals(ν2)))
    return sum(j -> Q.w[j] * f(Q.x[j], θ) * ν1[j] * ν2[j], jmin:jmax)
end

function ρIP(θ, nu1, k1, f, fnu2, g, gnu2, k2, ρ, Q::Nothing=nothing)
    return inner_product(nu1, k1, x -> f(x, θ), fnu2, x -> g(x, θ), gnu2, k2, ρ, int_order)
end

function ρIP(θ, nu1::Symbol, k1, f, fnu2::Symbol, g, gnu2::Symbol, k2, ρ, Q::QuadInfo)
    abs(k1 - k2) > 1 && return 0.0
    ν1 = get_nu(Q, nu1, k1)
    fν2 = get_nu(Q, fnu2, k2)
    gν2 = get_nu(Q, gnu2, k2)
    # fν2 and gν2 have the same nonzero values
    jmin = max(minimum(rowvals(ν1)), minimum(rowvals(fν2)))
    jmax = min(maximum(rowvals(ν1)), maximum(rowvals(fν2)))
    x = Q.x
    w = Q.w
    return sum(j -> w[j] * ν1[j] * (f(x[j], θ) * fν2[j] + g(x[j], θ) * gν2[j]), jmin:jmax)
end

function ρIP(θ, shot, sym, nu1, k1, fnu2, gnu2, k2, ρ, m, Q::Nothing=nothing)

    smt, cmt = sincos(m * θ)

    integrand(x, f, g) = nu1(x, k1, ρ) * (f * fnu2(x, k2, ρ) + g * gnu2(x, k2, ρ))

    I = 0.0
    if sym === :cs_ρρ_ρθ
        function int_cs_ρρ_ρθ(x)
            grr, grt = gρρ_gρθ(shot, x, θ)
            f = -cmt * grr
            g = m * smt * grt
            return integrand(x, f, g)
        end
        I = inner_product(int_cs_ρρ_ρθ, k1, k2, ρ, int_order)
    elseif sym === :cs_ρθ_θθ
        function int_cs_ρθ_θθ(x)
            grt, gtt = gρθ_gθθ(shot, x, θ)
            f = -cmt * grt
            g = m * smt * gtt
            return integrand(x, f, g)
        end
        I = inner_product(int_cs_ρθ_θθ, k1, k2, ρ, int_order)
    elseif sym === :sc_ρρ_ρθ
        function int_sc_ρρ_ρθ(x)
            grr, grt = gρρ_gρθ(shot, x, θ)
            f = -smt * grr
            g = -m * cmt * grt
            return integrand(x, f, g)
        end
        I = inner_product(int_sc_ρρ_ρθ, k1, k2, ρ, int_order)
    elseif sym === :sc_ρθ_θθ
        function int_sc_ρθ_θθ(x)
            grt, gtt = gρθ_gθθ(shot, x, θ)
            f = -smt * grt
            g = -m * cmt * gtt
            return integrand(x, f, g)
        end
        I = inner_product(int_sc_ρθ_θθ, k1, k2, ρ, int_order)
    end
    return I
end

function int_cs_ρρ_ρθ(j, l, m, ν1, fν2, gν2, Q)
    grr = Q.gρρ[j, l]
    grt = Q.gρθ[j, l]
    f = -(m == 0 ? 1.0 : Q.Fcos[m, l]) * grr
    g = m == 0 ? 0.0 : m * Q.Fsin[m, l] * grt
    return ν1[j] * (f * fν2[j] + g * gν2[j])
end

function int_cs_ρθ_θθ(j, l, m, ν1, fν2, gν2, Q)
    grt = Q.gρθ[j, l]
    gtt = Q.gθθ[j, l]
    f = -(m == 0 ? 1.0 : Q.Fcos[m, l]) * grt
    g = m == 0 ? 0.0 : m * Q.Fsin[m, l] * gtt
    return ν1[j] * (f * fν2[j] + g * gν2[j])
end

function int_sc_ρρ_ρθ(j, l, m, ν1, fν2, gν2, Q)
    m == 0 && return 0.0
    grr = Q.gρρ[j, l]
    grt = Q.gρθ[j, l]
    f = -Q.Fsin[m, l] * grr
    g = -m * Q.Fcos[m, l] * grt
    return ν1[j] * (f * fν2[j] + g * gν2[j])
end

function int_sc_ρθ_θθ(j, l, m, ν1, fν2, gν2, Q)
    m == 0 && return 0.0
    grt = Q.gρθ[j, l]
    gtt = Q.gθθ[j, l]
    f = -Q.Fsin[m, l] * grt
    g = -m * Q.Fcos[m, l] * gtt
    return ν1[j] * (f * fν2[j] + g * gν2[j])
end

function ρIP(θ, shot, sym, nu1::Symbol, k1, fnu2::Symbol, gnu2::Symbol, k2, ρ, m, Q::QuadInfo)

    abs(k1 - k2) > 1 && return 0.0
    ν1 = get_nu(Q, nu1, k1)
    fν2 = get_nu(Q, fnu2, k2)
    gν2 = get_nu(Q, gnu2, k2)

    l = MillerExtendedHarmonic.θindex(θ, Q.Fsin)

    jmin = max(minimum(rowvals(ν1)), minimum(rowvals(fν2)))
    jmax = min(maximum(rowvals(ν1)), maximum(rowvals(fν2)))

    I = 0.0
    if sym === :cs_ρρ_ρθ
        I = sum(j -> Q.w[j] *int_cs_ρρ_ρθ(j, l, m, ν1, fν2, gν2, Q), jmin:jmax)
    elseif sym === :cs_ρθ_θθ
        I = sum(j -> Q.w[j] *int_cs_ρθ_θθ(j, l, m, ν1, fν2, gν2, Q), jmin:jmax)
    elseif sym === :sc_ρρ_ρθ
        I = sum(j -> Q.w[j] *int_sc_ρρ_ρθ(j, l, m, ν1, fν2, gν2, Q), jmin:jmax)
    elseif sym === :sc_ρθ_θθ
        I = sum(j -> Q.w[j] *int_sc_ρθ_θθ(j, l, m, ν1, fν2, gν2, Q), jmin:jmax)
    end
    return I
end

# Fourier decomposition (all m values) of ρIP_f_nu
# Doing this for all k and nu will give 2D decomposition of f in to FEs for ρ and Fourier for θ
function θFD_ρIP_f_nu(f, nu, k, ρ, M; fft_op::Union{Nothing, Symbol}=nothing)
    g(θ) = ρIP_f_nu(θ, f, nu, k, ρ)
    return fourier_decompose(g, M; fft_op)
end

function θFD_ρIP_f_nu!(CS, f::F1, nu::F2, k, ρ, M, Fi, Fo, P, Q=nothing; reset_CS = false, fft_op::Union{Nothing, Symbol}=nothing)  where {F1, F2}
    return fourier_decompose!(CS, θ -> ρIP(θ, f, nu, k, ρ, Q), M, Fi, Fo, P, Q; reset_CS, fft_op)
end

function θFD_ρIP_f_nu_nu!(CS, f, nu1, k1, nu2, k2, ρ, M, Fi, Fo, P, Q=nothing; reset_CS = false, fft_op::Union{Nothing, Symbol}=nothing)
    return fourier_decompose!(CS, θ -> ρIP(θ, f, nu1, k1, nu2, k2, ρ, Q), M, Fi, Fo, P, Q; reset_CS, fft_op)
end

function θFD_ρIP!(CS, nu1, k1, f, fnu2, g, gnu2, k2, ρ, M, Fi, Fo, P, Q=nothing; reset_CS = false, fft_op::Union{Nothing, Symbol}=nothing)
    return fourier_decompose!(CS, θ -> ρIP(θ, nu1, k1, f, fnu2, g, gnu2, k2, ρ, Q), M, Fi, Fo, P, Q; reset_CS, fft_op)
end

function θFD_ρIP!(CS, shot::Shot, sym::Symbol, nu1, k1, fnu2, gnu2, k2, ρ, m, M, Fi, Fo, P, Q=nothing; reset_CS = false, fft_op::Union{Nothing, Symbol}=nothing)
    return fourier_decompose!(CS, θ -> ρIP(θ, shot, sym, nu1, k1, fnu2, gnu2, k2, ρ, m, Q), M, Fi, Fo, P, Q; reset_CS, fft_op)
end

function compute_element(CS::AbstractVector{<:Real}, shot::Shot, Ftype::Symbol, m, nu1_type, k1, nu2_type, k2, ρ,  M, Fi, dFi, Fo, P, Q=nothing; reset_CS = false)
    return dual_fourier_decompose!(CS, θ -> dual_ρIP(θ, shot, Ftype, m, nu1_type, k1, nu2_type, k2, ρ, Q), M, Fi, dFi, Fo, P, Q; reset_CS)
end

function int_ab(x, fa, ga, fb, gb, nu1, D_nu1, nu2, D_nu2)
    Dn2 = D_nu2(x)
    n2  = nu2(x)
    I1 = D_nu1(x) * (fa * Dn2 + ga * n2)
    I2 = nu1(x) * (fb * Dn2 + gb * n2)
    return SVector(I1, I2)
end

function int_cos(shot::Shot, x::Real, θ::Real, ncmt, msmt, nu1, D_nu1, nu2, D_nu2)
    grr, grt, gtt = gρρ_gρθ_gθθ(shot, x, θ)
    f = ncmt * grr
    g = msmt * grt
    df = ncmt * grt
    dg = msmt * gtt
    return int_ab(x, f, g, df, dg, nu1, D_nu1, nu2, D_nu2)
end

function int_sin(shot::Shot, x::Real, θ::Real, nsmt, nmcmt, nu1, D_nu1, nu2, D_nu2)
    grr, grt, gtt = gρρ_gρθ_gθθ(shot, x, θ)
    f = nsmt * grr
    g = nmcmt * grt
    df = nsmt * grt
    dg = nmcmt * gtt
    return int_ab(x, f, g, df, dg, nu1, D_nu1, nu2, D_nu2)
end

function dual_ρIP(θ, shot, Ftype::Symbol, m::Integer, ν1_type::Symbol, k1, ν2_type::Symbol, k2, ρ, Q::Nothing=nothing)

    nu1   = x -> (ν1_type === :odd) ? νo(x, k1, ρ)   : νe(x, k1, ρ)
    D_nu1 = x -> (ν1_type === :odd) ? D_νo(x, k1, ρ) : D_νe(x, k1, ρ)
    nu2   = x -> (ν2_type === :odd) ? νo(x, k2, ρ)   : νe(x, k2, ρ)
    D_nu2 = x -> (ν2_type === :odd) ? D_νo(x, k2, ρ) : D_νe(x, k2, ρ)

    smt, cmt = sincos(m * θ)

    I  = 0.0
    dI = 0.0
    if Ftype === :cos
        ncmt = -cmt
        msmt = m * smt
        I, dI = dual_inner_product(x -> int_cos(shot, x, θ, ncmt, msmt, nu1, D_nu1, nu2, D_nu2), k1, k2, ρ, int_order)
    elseif Ftype === :sin
        nsmt = -smt
        nmcmt = -m * cmt
        I, dI = dual_inner_product(x -> int_sin(shot, x, θ, nsmt, nmcmt, nu1, D_nu1, nu2, D_nu2), k1, k2, ρ, int_order)
    end
    return I, dI
end

function int_cos(j::Int, l::Int, ncmt, msmt, ν1, D_ν1, ν2, D_ν2, Q::QuadInfo)
    grr = Q.gρρ[j, l]
    grt = Q.gρθ[j, l]
    gtt = Q.gθθ[j, l]
    f = ncmt * grr
    g = msmt * grt
    df = ncmt * grt
    dg = msmt * gtt
    I1 = D_ν1[j] * (f * D_ν2[j] + g * ν2[j])
    I2 = ν1[j] * (df * D_ν2[j] + dg * ν2[j])
    return SVector(I1, I2)
end

function int_sin(j::Int, l::Int, nsmt, nmcmt, ν1, D_ν1, ν2, D_ν2, Q::QuadInfo)
    grr = Q.gρρ[j, l]
    grt = Q.gρθ[j, l]
    gtt = Q.gθθ[j, l]
    f = nsmt * grr
    g = nmcmt * grt
    df = nsmt * grt
    dg = nmcmt * gtt
    I1 = D_ν1[j] * (f * D_ν2[j] + g * ν2[j])
    I2 = ν1[j] * (df * D_ν2[j] + dg * ν2[j])
    return SVector(I1, I2)
end

function dual_setup(θ, m::Integer, ν1_type::Symbol, k1, ν2_type::Symbol, k2, Q::QuadInfo)
    D_ν1_type = (ν1_type === :odd) ? :D_odd : :D_even
    ν1   = get_nu(Q, ν1_type, k1)
    D_ν1 = get_nu(Q, D_ν1_type, k1)

    D_ν2_type = (ν2_type === :odd) ? :D_odd : :D_even
    ν2   = get_nu(Q, ν2_type, k2)
    D_ν2 = get_nu(Q, D_ν2_type, k2)

    l = MillerExtendedHarmonic.θindex(θ, Q.Fsin)

    smt = (m == 0) ? 0.0 : Q.Fsin[m, l]
    cmt = (m == 0) ? 1.0 : Q.Fcos[m, l]

    jmin = max(minimum(rowvals(ν1)), minimum(rowvals(ν2)))
    jmax = min(maximum(rowvals(ν1)), maximum(rowvals(ν2)))
    return ν1, D_ν1, ν2, D_ν2, l, smt, cmt, jmin, jmax
end

function dual_ρIP(θ, shot, Ftype::Symbol, m::Integer, ν1_type::Symbol, k1, ν2_type::Symbol, k2, ρ, Q::QuadInfo)
    if Ftype === :cos
        I, dI = dual_ρIP_cos(θ, m, ν1_type, k1, ν2_type, k2, Q)
    elseif Ftype === :sin
        I, dI = dual_ρIP_sin(θ, m, ν1_type, k1, ν2_type, k2, Q)
    end
    return I, dI
end

function dual_ρIP_cos(θ, m::Integer, ν1_type::Symbol, k1, ν2_type::Symbol, k2, Q::QuadInfo)

    abs(k1 - k2) > 1 && return 0.0, 0.0

    ν1, D_ν1, ν2, D_ν2, l, smt, cmt, jmin, jmax = dual_setup(θ, m, ν1_type, k1, ν2_type, k2, Q)

    ncmt = -cmt
    msmt = m * smt
    I = 0.0
    dI = 0.0
    for j in jmin:jmax
        i, di = Q.w[j] .* int_cos(j, l, ncmt, msmt, ν1, D_ν1, ν2, D_ν2, Q)
        I += i
        dI += di
    end
    return I, dI
end

function dual_ρIP_sin(θ, m::Integer, ν1_type::Symbol, k1, ν2_type::Symbol, k2, Q::QuadInfo)

    abs(k1 - k2) > 1 && return 0.0, 0.0

    ν1, D_ν1, ν2, D_ν2, l, smt, cmt, jmin, jmax = dual_setup(θ, m, ν1_type, k1, ν2_type, k2, Q)

    I  = 0.0
    dI = 0.0
    nsmt = -smt
    nmcmt = -m * cmt
    I  = 0.0
    dI = 0.0
    for j in jmin:jmax
        i, di = Q.w[j] .* int_sin(j, l, nsmt, nmcmt, ν1, D_ν1, ν2, D_ν2, Q)
        I += i
        dI += di
    end
    return I, dI
end