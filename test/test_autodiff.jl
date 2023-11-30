using TEQUILA, MillerExtendedHarmonic, ForwardDiff, Test

shotd3d = Shot(11, 11, 7, "../sample/g_chease_mxh_d3d")
bnd = shotd3d.surfaces[:,end]
dP_dψ = shotd3d.dP_dψ
F_dF_dψ = shotd3d.F_dF_dψ
Pbnd = shotd3d.Pbnd
Fbnd = shotd3d.Fbnd

function solve_from_bnd(bnd)
    # initialize TEQUILA
    shot = Shot(11, 11, MXH(bnd); dP_dψ, F_dF_dψ, Pbnd, Fbnd);

    # solve TEQUILA equilibrium
    #refill = solve(shot, 3; debug=false)

    return shot.surfaces
end


@time refill = solve_from_bnd(bnd); # runs fine
res = ForwardDiff.jacobian(solve_from_bnd, bnd);
@test !isnothing(res)


#=
# Scratch code
N = 11; M = 11; boundary = MXH(bnd); Raxis = boundary.R0; Zaxis = boundary.Z0; P = nothing;
ρ = range(0, 1, N)

L = length(boundary.c)
surfaces = make_surfaces(boundary, ρ; Raxis, Zaxis)

T = ForwardDiff.Dual{ForwardDiff.Tag{typeof(solve_from_bnd), Float64}, Float64, 10}
vec = fill(zero(T), 10)
=#
