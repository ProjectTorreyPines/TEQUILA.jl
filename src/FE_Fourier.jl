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

function fourier_decompose!(CS::AbstractVector{<:Real}, f, M::Integer,
                            Fi::AbstractVector{<:Complex}, Fo::AbstractVector{<:Complex}, P::FFTW.FFTWPlan;
                            reset_CS = false, fft_op::Union{Nothing, Symbol}=nothing)
    invM2 = 1.0 / (M + 2)
    x = range(0,twopi, 2M+5)[1:end-1]
    for (k, θ) in enumerate(x)
        Fi[k] = f(θ)
    end
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
    #@assert !any(isnan.(CS))
    return CS
end

function dual_fourier_decompose!(CS::AbstractVector{<:Real}, f, M::Integer,
                            Fi::AbstractVector{<:Complex}, dFi::AbstractVector{<:Complex}, Fo::AbstractVector{<:Complex}, P::FFTW.FFTWPlan;
                            reset_CS = false)

    reset_CS && (CS .= 0.0)

    invM2 = 1.0 / (M + 2)
    x = range(0, twopi, 2M+5)[1:end-1]

    for (k, θ) in enumerate(x)
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
@inline ρIP(θ, f, nu, k, ρ) = inner_product(x -> f(x, θ), nu, k, ρ, 5)

@inline ρIP(θ, f, nu1, k1, nu2, k2, ρ) = inner_product(x -> f(x, θ), nu1, k1, nu2, k2, ρ, 5)

function ρIP(θ, nu1, k1, f, fnu2, g, gnu2, k2, ρ)
    return inner_product(nu1, k1, x -> f(x, θ), fnu2, x -> g(x, θ), gnu2, k2, ρ, 5)
end

function ρIP(θ, shot, sym, nu1, k1, fnu2, gnu2, k2, ρ, m)

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
        I = inner_product(int_cs_ρρ_ρθ, k1, k2, ρ, 5)
    elseif sym === :cs_ρθ_θθ
        function int_cs_ρθ_θθ(x)
            grt, gtt = gρθ_gθθ(shot, x, θ)
            f = -cmt * grt
            g = m * smt * gtt
            return integrand(x, f, g)
        end
        I = inner_product(int_cs_ρθ_θθ, k1, k2, ρ, 5)
    elseif sym === :sc_ρρ_ρθ
        function int_sc_ρρ_ρθ(x)
            grr, grt = gρρ_gρθ(shot, x, θ)
            f = -smt * grr
            g = -m * cmt * grt
            return integrand(x, f, g)
        end
        I = inner_product(int_sc_ρρ_ρθ, k1, k2, ρ, 5)
    elseif sym === :sc_ρθ_θθ
        function int_sc_ρθ_θθ(x)
            grt, gtt = gρθ_gθθ(shot, x, θ)
            f = -smt * grt
            g = -m * cmt * gtt
            return integrand(x, f, g)
        end
        I = inner_product(int_sc_ρθ_θθ, k1, k2, ρ, 5)
    end
    return I
end

# Fourier decomposition (all m values) of ρIP_f_nu
# Doing this for all k and nu will give 2D decomposition of f in to FEs for ρ and Fourier for θ
function θFD_ρIP_f_nu(f, nu, k, ρ, M; fft_op::Union{Nothing, Symbol}=nothing)
    g(θ) = ρIP_f_nu(θ, f, nu, k, ρ)
    return fourier_decompose(g, M; fft_op)
end

function θFD_ρIP_f_nu!(CS, f, nu, k, ρ, M, Fi, Fo, P; reset_CS = false, fft_op::Union{Nothing, Symbol}=nothing)
    return fourier_decompose!(CS, θ -> ρIP(θ, f, nu, k, ρ), M, Fi, Fo, P; reset_CS, fft_op)
end

function θFD_ρIP_f_nu_nu!(CS, f, nu1, k1, nu2, k2, ρ, M, Fi, Fo, P; reset_CS = false, fft_op::Union{Nothing, Symbol}=nothing)
    return fourier_decompose!(CS, θ -> ρIP(θ, f, nu1, k1, nu2, k2, ρ), M, Fi, Fo, P; reset_CS, fft_op)
end

function θFD_ρIP!(CS, nu1, k1, f, fnu2, g, gnu2, k2, ρ, M, Fi, Fo, P; reset_CS = false, fft_op::Union{Nothing, Symbol}=nothing)
    return fourier_decompose!(CS, θ -> ρIP(θ, nu1, k1, f, fnu2, g, gnu2, k2, ρ), M, Fi, Fo, P; reset_CS, fft_op)
end

function θFD_ρIP!(CS, shot::Shot, sym::Symbol, nu1, k1, fnu2, gnu2, k2, ρ, m, M, Fi, Fo, P; reset_CS = false, fft_op::Union{Nothing, Symbol}=nothing)
    return fourier_decompose!(CS, θ -> ρIP(θ, shot, sym, nu1, k1, fnu2, gnu2, k2, ρ, m), M, Fi, Fo, P; reset_CS, fft_op)
end

function θFD_ρIP_2!(CS, shot::Shot, sym::Symbol, dsym::Symbol, nu1, k1, fnu2, gnu2, dfnu2, dgnu2, k2, ρ, m, M, Fi, Fo, P; reset_CS = false)
    return fourier_decompose!(CS, dCS, θ -> ρIP(θ, shot, sym, nu1, k1, fnu2, gnu2, k2, ρ, m), M, Fi, Fo, P; reset_CS, fft_op)
end

#@views θFD_ρIP!(Astar[Jes, Ie + Mc], shot, :cs_ρρ_ρθ, D_νe, j, D_νo, νo, j-1, ρ, m, M, Fi, Fo, P)
#@views θFD_ρIP!(Astar[Jes, Ie + Mc], shot, :cs_ρθ_θθ, νe, j, D_νo, νo, j-1, ρ, m, M, Fi, Fo, P, fft_op=:derivative)

function compute_element(CS::AbstractVector{<:Real}, shot::Shot, Ftype::Symbol, m, nu1_type, k1, nu2_type, k2, ρ,  M, Fi, dFi, Fo, P; reset_CS = false)
    return dual_fourier_decompose!(CS, θ -> dual_ρIP(θ, shot, Ftype, m, nu1_type, k1, nu2_type, k2, ρ), M, Fi, dFi, Fo, P; reset_CS)
end

function int_ab(x, fa, ga, fb, gb, nu1, D_nu1, nu2, D_nu2)
    Dn2 = D_nu2(x)
    n2  = nu2(x)
    I1 = D_nu1(x) * (fa * Dn2 + ga * n2)
    I2 = nu1(x) * (fb * Dn2 + gb * n2)
    return SVector(I1, I2)
end

function dual_ρIP(θ, shot, Ftype::Symbol, m::Integer, ν1_type::Symbol, k1, ν2_type::Symbol, k2, ρ)

    nu1(x)   = (ν1_type === :odd) ? νo(x, k1, ρ)   : νe(x, k1, ρ)
    D_nu1(x) = (ν1_type === :odd) ? D_νo(x, k1, ρ) : D_νe(x, k1, ρ)
    nu2(x)   = (ν2_type === :odd) ? νo(x, k2, ρ)   : νe(x, k2, ρ)
    D_nu2(x) = (ν2_type === :odd) ? D_νo(x, k2, ρ) : D_νe(x, k2, ρ)

    #int1(x, f, g) = D_nu1(x) * (f * D_nu2(x) + g * nu2(x))
    #int2(x, f, g) = nu1(x) * (f * D_nu2(x) + g * nu2(x))

    smt, cmt = sincos(m * θ)

    I  = 0.0
    dI = 0.0
    if Ftype === :cos
        ncmt = -cmt
        msmt = m * smt
        function int_cos(x)
            grr, grt, gtt = gρρ_gρθ_gθθ(shot, x, θ)
            f = ncmt * grr
            g = msmt * grt
            df = ncmt * grt
            dg = msmt * gtt
            return int_ab(x, f, g, df, dg, nu1, D_nu1, nu2, D_nu2)
        end
        I, dI = dual_inner_product(int_cos, k1, k2, ρ, 5)
    elseif Ftype === :sin
        nsmt = -smt
        nmcmt = -m * cmt
        function int_sin(x)
            grr, grt, gtt = gρρ_gρθ_gθθ(shot, x, θ)
            f = nsmt * grr
            g = nmcmt * grt
            df = nsmt * grt
            dg = nmcmt * gtt
            return int_ab(x, f, g, df, dg, nu1, D_nu1, nu2, D_nu2)
        end
        I, dI = dual_inner_product(int_sin, k1, k2, ρ, 5)
    end
    return I, dI
end