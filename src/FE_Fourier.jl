function fft_prealloc(M::Integer)
    tmp = zeros(2M + 4)
    Fi = complex(tmp)
    dFi = complex(tmp)
    Fo = complex(tmp)
    P = plan_fft(Fi)
    return Fi, dFi, Fo, P
end

function fft_prealloc_threaded(M::Integer)
    tmp = zeros(2M + 4)
    Fis = [complex(tmp) for _ in 1:Threads.nthreads()]
    dFis = [complex(tmp) for _ in 1:Threads.nthreads()]
    Fos = [complex(tmp) for _ in 1:Threads.nthreads()]
    Ps = [plan_fft(complex(tmp)) for _ in 1:Threads.nthreads()]
    return Fis, dFis, Fos, Ps
end

function fourier_decompose!(
    CS::AbstractVector{<:Real},
    f::F1,
    M::Integer,
    Fi::AbstractVector{<:Complex},
    Fo::AbstractVector{<:Complex},
    P::FFTW.FFTWPlan,
    Q::QuadInfo;
    reset_CS=false,
    fft_op::Union{Nothing,Symbol}=nothing
) where {F1<:Function}
    return fourier_decompose!(CS, f, M, Fi, Fo, P, Q.θ; reset_CS, fft_op)
end

function fourier_decompose!(
    CS::AbstractVector{<:Real},
    f::F1,
    M::Integer,
    Fi::AbstractVector{<:Complex},
    Fo::AbstractVector{<:Complex},
    P::FFTW.FFTWPlan,
    θs::AbstractVector{<:Real};
    reset_CS=false,
    fft_op::Union{Nothing,Symbol}=nothing
) where {F1<:Function}
    invM2 = 1.0 / (M + 2)
    @. Fi = f(θs)
    mul!(Fo, P, Fi)
    reset_CS && (CS .= 0.0)
    if fft_op === :derivative
        m_M2 = range(0, M * invM2, M + 1)
        @views CS[1:2:end] .+= m_M2 .* imag.(Fo[1:(M+1)])
        @views CS[2:2:end] .+= m_M2[2:end] .* real.(Fo[2:(M+1)])
    else
        Fo[1] *= 0.5
        @views CS[1:2:end] .+= real.(Fo[1:(M+1)]) .* invM2
        @views CS[2:2:end] .+= .-imag.(Fo[2:(M+1)]) .* invM2 # fft sign convention
    end
    return CS
end

function dual_fourier_decompose!(
    CS::AbstractVector{<:Real},
    f,
    M::Integer,
    Fi::AbstractVector{<:Complex},
    dFi::AbstractVector{<:Complex},
    Fo::AbstractVector{<:Complex},
    P::FFTW.FFTWPlan,
    Q::QuadInfo;
    reset_CS=false
)
    return dual_fourier_decompose!(CS, f, M, Fi, dFi, Fo, P, Q.θ; reset_CS)
end

function dual_fourier_decompose!(CS::AbstractVector{<:Real}, f, M::Integer, Fi::AbstractVector{<:Complex}, dFi::AbstractVector{<:Complex}, Fo::AbstractVector{<:Complex},
    P::FFTW.FFTWPlan, θs::AbstractVector{<:Real};
    reset_CS=false)
    reset_CS && (CS .= 0.0)
    invM2 = 1.0 / (M + 2)
    M1 = M + 1

    @assert length(θs) == length(Fi) == length(dFi) # this syntax works!
    @inbounds for (k, θ) in enumerate(θs)
        Fi[k], dFi[k] = f(θ)
    end

    mul!(Fo, P, Fi)
    Fo[1] *= 0.5
    @assert length(Fo) >= M1
    @views @inbounds CS[1:2:end] .+= real.(Fo[1:M1]) .* invM2
    @views @inbounds CS[2:2:end] .-= imag.(Fo[2:M1]) .* invM2 # fft sign convention

    mul!(Fo, P, dFi)
    m_M2 = range(0, M * invM2, M + 1)
    @views @inbounds CS[1:2:end] .+= m_M2 .* imag.(Fo[1:M1])
    @views @inbounds CS[2:2:end] .+= m_M2[2:end] .* real.(Fo[2:M1])

    return CS
end

# At fixed θ, give inner product of f(x,θ) and the basis nu(x,k,ρ)

function ρIP(θ, f, nu::Symbol, k, Q::QuadInfo)
    ν = get_nu(Q, nu, k)
    return sum(j -> Q.w[j] * f(Q.x[j], θ) * ν[j], rowvals(ν))
end

# Fourier decomposition (all m values) of ρIP_f_nu
# Doing this for all k and nu will give 2D decomposition of f in to FEs for ρ and Fourier for θ

function θFD_ρIP_f_nu!(CS, f::F1, nu::F2, k, M, Fi, Fo, P, Q=nothing; reset_CS=false, fft_op::Union{Nothing,Symbol}=nothing) where {F1,F2}
    return fourier_decompose!(CS, θ -> ρIP(θ, f, nu, k, Q), M, Fi, Fo, P, Q; reset_CS, fft_op)
end

function compute_element_cos(CS::AbstractVector{<:Real}, m, nu1_type, k1, nu2_type, k2, M, Fi, dFi, Fo, P, Q=nothing; reset_CS=false)
    return dual_fourier_decompose!(CS, θ -> dual_ρIP_cos(θ, m, nu1_type, k1, nu2_type, k2, Q), M, Fi, dFi, Fo, P, Q; reset_CS)
end

function compute_element_sin(CS::AbstractVector{<:Real}, m, nu1_type, k1, nu2_type, k2, M, Fi, dFi, Fo, P, Q=nothing; reset_CS=false)
    return dual_fourier_decompose!(CS, θ -> dual_ρIP_sin(θ, m, nu1_type, k1, nu2_type, k2, Q), M, Fi, dFi, Fo, P, Q; reset_CS)
end

function int_setup(j::Int, l::Int, ν1, D_ν1, ν2, D_ν2, Q::QuadInfo)
    grr = Q.gρρ[j, l]
    grt = Q.gρθ[j, l]
    gtt = Q.gθθ[j, l]
    nu1 = ν1[j]
    D_nu1 = D_ν1[j]
    nu2 = ν2[j]
    D_nu2 = D_ν2[j]
    return grr, grt, gtt, nu1, D_nu1, nu2, D_nu2
end

function int_cos(j::Int, l::Int, ncmt, msmt, ν1, D_ν1, ν2, D_ν2, Q::QuadInfo)
    grr, grt, gtt, nu1, D_nu1, nu2, D_nu2 = int_setup(j, l, ν1, D_ν1, ν2, D_ν2, Q)
    f = ncmt * D_nu2
    g = msmt * nu2
    I1 = D_nu1 * (f * grr + g * grt)
    I2 = nu1 * (f * grt + g * gtt)
    return SVector(I1, I2)
end

function int_sin(j::Int, l::Int, nsmt, nmcmt, ν1, D_ν1, ν2, D_ν2, Q::QuadInfo)
    grr, grt, gtt, nu1, D_nu1, nu2, D_nu2 = int_setup(j, l, ν1, D_ν1, ν2, D_ν2, Q)
    f = nsmt * D_nu2
    g = nmcmt * nu2
    I1 = D_nu1 * (f * grr + g * grt)
    I2 = nu1 * (f * grt + g * gtt)
    return SVector(I1, I2)
end

function dual_setup(θ, m::Integer, ν1_type::Symbol, k1, ν2_type::Symbol, k2, Q::QuadInfo)
    D_ν1_type = (ν1_type === :odd) ? :D_odd : :D_even
    ν1 = get_nu(Q, ν1_type, k1)
    D_ν1 = get_nu(Q, D_ν1_type, k1)

    D_ν2_type = (ν2_type === :odd) ? :D_odd : :D_even
    ν2 = get_nu(Q, ν2_type, k2)
    D_ν2 = get_nu(Q, D_ν2_type, k2)

    l = MillerExtendedHarmonic.θindex(θ, Q.Fsin)

    smt = (m == 0) ? 0.0 : Q.Fsin[m, l]
    cmt = (m == 0) ? 1.0 : Q.Fcos[m, l]

    jmin = max(minimum(rowvals(ν1)), minimum(rowvals(ν2)))
    jmax = min(maximum(rowvals(ν1)), maximum(rowvals(ν2)))
    return ν1, D_ν1, ν2, D_ν2, l, smt, cmt, jmin, jmax
end

function dual_ρIP_cos(θ, m::Integer, ν1_type::Symbol, k1, ν2_type::Symbol, k2, Q::QuadInfo)
    abs(k1 - k2) > 1 && return 0.0, 0.0
    ν1, D_ν1, ν2, D_ν2, l, smt, cmt, jmin, jmax = dual_setup(θ, m, ν1_type, k1, ν2_type, k2, Q)
    ncmt = -cmt
    msmt = m * smt
    return sum(j -> Q.w[j] .* int_cos(j, l, ncmt, msmt, ν1, D_ν1, ν2, D_ν2, Q), jmin:jmax)
end

function dual_ρIP_sin(θ, m::Integer, ν1_type::Symbol, k1, ν2_type::Symbol, k2, Q::QuadInfo)
    abs(k1 - k2) > 1 && return 0.0, 0.0
    ν1, D_ν1, ν2, D_ν2, l, smt, cmt, jmin, jmax = dual_setup(θ, m, ν1_type, k1, ν2_type, k2, Q)
    nsmt = -smt
    nmcmt = -m * cmt
    return sum(j -> Q.w[j] .* int_sin(j, l, nsmt, nmcmt, ν1, D_ν1, ν2, D_ν2, Q), jmin:jmax)
end