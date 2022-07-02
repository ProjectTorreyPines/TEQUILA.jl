# Ψ constant on Miller shape and equal to ρ^2
function Ψmiller(R::Real, Z::Real; R₀::Real=0.0, Z₀::Real=0.0, a::Real=1.0, κ::Real=1.0, δ::Real=0.0)
    function Δ!(F, x)
        F[1] = R₀ + a * x[1] * cos(x[2] + asin(δ) * sin(x[2])) - R
        F[2] = Z₀ + a * κ * x[1] * sin(x[2]) - Z
    end
    S =  NLsolve.nlsolve(Δ!, [sqrt(((R-R₀)/a)^2 + ((Z-Z₀)/(κ*a))^2),atan(Z-Z₀, R-R₀)])
    if NLsolve.converged(S)
        return S.zero[1]^2
    else
        error("Did not converge")
    end
end

function concentric_surfaces(ρ::AbstractVector{<:Real}, boundary::MXH)
    surfaces = Vector{MXH}(undef, N)
    for (i, x) in enumerate(ρ)
        surfaces[i] = deepcopy(boundary)
        # these go to zero as you go to axis
        surfaces[i].ϵ *= x
        surfaces[i].c *= x
        surfaces[i].s *= x
        # the rest stay constant as a first guess
    end
    # θ = range(0, 2π, 1001)
    # R = zero(θ)
    # Z = zero(θ)
    # p = plot(aspect_ratio=:equal)
    # for surface in surfaces
    #     for (i, t) in enumerate(θ)
    #         R[i], Z[i] = surface(t)
    #     end
    #     plot!(p, R, Z, c=:black)
    # end
    # display(p)
    return surfaces
end

function first_shot(Ψ, boundary::MXH, N, M)

    ρ = range(0, 1, N)

    surfaces = concentric_surfaces(ρ, boundary)

    # Now compute C matrix for input Ψ(ρ, θ)
    #C = zeros(2N, 2M)
    #A = zeros(4*N*M, 4*N*M)
    #B = zeros(4*N*M)

    


    #shot = TEQUILAEquilibrium(ρ, surfaces, C)

    return# shot
end