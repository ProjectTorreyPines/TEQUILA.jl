"""
    Ψmiller(R::Real, Z::Real; R0::Real=0.0, Z0::Real=0.0, a::Real=1.0, κ::Real=1.0, δ::Real=0.0)

Compute flux equal to ρ^2 at point (R, Z), assuming normalized radius inside concentric Miller shape
"""
function Ψmiller(R::Real, Z::Real; R0::Real=0.0, Z0::Real=0.0, a::Real=1.0, κ::Real=1.0, δ::Real=0.0)
    function Δ!(F, x)
        F[1] = R0 + a * x[1] * cos(x[2] + asin(δ) * sin(x[2])) - R
        return F[2] = Z0 + a * κ * x[1] * sin(x[2]) - Z
    end
    S = NLsolve.nlsolve(Δ!, [sqrt(((R - R0) / a)^2 + ((Z - Z0) / (κ * a))^2), -atan(Z - Z0, R - R0)])
    if NLsolve.converged(S)
        return S.zero[1]^2
    else
        error("Did not converge")
    end
end