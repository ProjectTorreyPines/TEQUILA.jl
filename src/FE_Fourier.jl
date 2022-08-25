f_nu_mu(x, θ, f, nu, k, mu, m, ρ) = f(x, θ) * nu(x, k, ρ) * mu(m*θ)

"""
    fourier_decompose(f, M::Integer)

Decompose f(θ) into a Fourier series with m from 0 to M
Returns cosine coeffients and sine coefficients as tuple
"""
function fourier_decompose(f, M::Integer)
    CS = zeros(2 * (M + 1))
    return fourier_decompose!(CS, f, M)
end

function fourier_decompose!(CS::AbstractVector{<:Real}, f, M::Integer)
    invM2 = 1.0 / (M + 2)
    Δθ = π * invM2
    x = 0.0:Δθ:(2π-Δθ)
    y = [Complex(f(θ)) for θ in x]
    fft!(y)
    @views CS[1:2:end] .=  real.(y[1:(M+1)]) .* invM2
    @views CS[2:2:end] .= -imag.(y[1:(M+1)]) .* invM2 # fft sign convention
    @views CS[1] *= 0.5
    CS[2] = 0.0
    return CS #collect(Iterators.flatten(zip(Cc, Sc)))
end

# At fixed θ, give inner product of f(x,θ) and the basis nu(x,k,ρ)
ρIP_f_nu(θ, f, nu, k, ρ) = inner_product(x -> f(x,θ), nu, k, ρ, 5)

# Fourier decomposition (all m values) of ρIP_f_nu
# Doing this for all k and nu will give 2D decomposition of f in to FEs for ρ and Fourier for θ
θFD_ρIP_f_nu(f, nu, k, ρ, M) = fourier_decompose(θ -> ρIP_f_nu(θ, f, nu, k, ρ), M)

θFD_ρIP_f_nu!(CS, f, nu, k, ρ, M) = fourier_decompose!(CS, θ -> ρIP_f_nu(θ, f, nu, k, ρ), M)