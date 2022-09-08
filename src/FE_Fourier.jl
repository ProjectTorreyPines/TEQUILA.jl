f_nu_mu(x, θ, f, nu, k, mu, m, ρ) = f(x, θ) * nu(x, k, ρ) * mu(m*θ)

"""
    fourier_decompose(f, M::Integer)

Decompose f(θ) into a Fourier series with m from 0 to M
Returns cosine coeffients and sine coefficients as tuple
"""
function fft_prealloc(M::Integer)
    ctype = typeof(Complex(0.0))
    Fi = zeros(ctype, 2M+4)
    Fo = Vector{ctype}(undef, 2M+4)
    P = plan_fft(Fi)
    return Fi, Fo, P
end

function fourier_decompose(f, M::Integer; derivative=false)
    CS = zeros(2M + 1)
    Fi, Fo, P = fft_prealloc(M)
    return fourier_decompose!(CS, f, M, Fi, Fo, P; fft_op)
end

function fourier_decompose!(CS::AbstractVector{<:Real}, f, M::Integer,
                            Fi::AbstractVector{<:Complex}, Fo::AbstractVector{<:Complex}, P::FFTW.FFTWPlan;
                            reset_CS = false, fft_op::Union{Nothing, Symbol}=nothing)
    invM2 = 1.0 / (M + 2)
    Δθ = π * invM2
    x = 0.0:Δθ:(2π-Δθ)
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

# At fixed θ, give inner product of f(x,θ) and the basis nu(x,k,ρ)
@inline ρIP_f_nu(θ, f, nu, k, ρ) = inner_product(x -> f(x, θ), nu, k, ρ, 5)

@inline ρIP_f_nu_nu(θ, f, nu1, k1, nu2, k2, ρ) = inner_product(x -> f(x, θ), nu1, k1, nu2, k2, ρ, 5)

# Fourier decomposition (all m values) of ρIP_f_nu
# Doing this for all k and nu will give 2D decomposition of f in to FEs for ρ and Fourier for θ
function θFD_ρIP_f_nu(f, nu, k, ρ, M; fft_op::Union{Nothing, Symbol}=nothing)
    g(θ) = ρIP_f_nu(θ, f, nu, k, ρ)
    return fourier_decompose(g, M; fft_op)
end

function θFD_ρIP_f_nu!(CS, f, nu, k, ρ, M, Fi, Fo, P; reset_CS = false, fft_op::Union{Nothing, Symbol}=nothing)
    return fourier_decompose!(CS, θ -> ρIP_f_nu(θ, f, nu, k, ρ), M, Fi, Fo, P; reset_CS, fft_op)
end

function θFD_ρIP_f_nu_nu!(CS, f, nu1, k1, nu2, k2, ρ, M, Fi, Fo, P; reset_CS = false, fft_op::Union{Nothing, Symbol}=nothing)
    return fourier_decompose!(CS, θ -> ρIP_f_nu_nu(θ, f, nu1, k1, nu2, k2, ρ), M, Fi, Fo, P; reset_CS, fft_op)
end