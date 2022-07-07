# Ψ constant on Miller shape and equal to ρ^2
function Ψmiller(R::Real, Z::Real; R0::Real=0.0, Z0::Real=0.0, a::Real=1.0, κ::Real=1.0, δ::Real=0.0)
    function Δ!(F, x)
        F[1] = R0 + a * x[1] * cos(x[2] + asin(δ) * sin(x[2])) - R
        F[2] = Z0 + a * κ * x[1] * sin(x[2]) - Z
    end
    S =  NLsolve.nlsolve(Δ!, [sqrt(((R-R0)/a)^2 + ((Z-Z0)/(κ*a))^2),atan(Z-Z0, R-R0)])
    if NLsolve.converged(S)
        return S.zero[1]^2
    else
        error("Did not converge")
    end
end

function first_shot(Ψ, boundary::MXH, N, M)

    ρ = range(0, 1, N)

    surfaces = concentric_surface.(ρ, Ref(boundary))

    function Ψ_fc(ρ, θ)
        S = concentric_surface(ρ, boundary)
        return Ψ(S(θ)...)
    end

    # Now compute C matrix for input Ψ(ρ, θ)
    B = zeros(2*N,2(M+1))
    for j in 1:N
        B[2j-1,:] = θFD_ρIP_f_nu(Ψ_fc, νo, j, ρ, M)
        B[2j  ,:] = θFD_ρIP_f_nu(Ψ_fc, νe, j, ρ, M)
    end

    A = mass_matrix(N, ρ)

    C = zeros(2*N,2(M+1))
    for k in 1:(2(M+1))
        C[:,k] = A \ B[:,k]
    end

    return Shot(N, M, ρ, surfaces, C)
end