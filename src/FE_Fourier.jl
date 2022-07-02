f_nu_mu(x, θ, f, nu, k, mu, m, ρ) = f(x, θ) * nu(x, k, ρ) * mu(m*θ)

"""
    fourier_decompose(f, M::Integer)

Decompose f(θ) into a Fourier series with m from 0 to M
Returns cosine coeffients and sine coefficients as tuple
"""
function fourier_decompose(f, M::Integer)
    Δθ = π / (M+2)
    x = 0.0:Δθ:(2π-Δθ)
    y = [f(θ) for θ in x]
    Fy = fft(y)[1:(M+1)]
    Cc =  real.(Fy) / (M + 2)
    Sc = -imag.(Fy) / (M + 2)  # fft sign convention
    Cc[1] = 0.5 * Cc[1]
    Sc[1] = 0.0
    return Cc, Sc
end

# Not sure we'll need this
# Probably just fourier_decompose ρ inner products
function inner_product(f, nu, k, ρ, mu, m)
    Ix(θ) = mu(m*θ) * inner_product(x -> f(x,θ), nu, k, ρ)
    return @trapz range(0, 2π, 4m+1) θ Ix(θ)
end

#function inner_product(f, nu, k, ρ, mu, m, N)
#    Iθ(x) = @trapz range(0, 2π, 4m+1) θ f(x,θ) * mu(m*θ)
#    return @trapz range(0, 2π, 2N+1) θ Ix(θ)
#end