using TEQUILA
using Test

P(x) = 1e5 * (1 - x)
Pp(x) = -1e5 * (1 - x ^ 2)
FFp(x) = 3.0 * (1 - x ^ 2)
J(x) = 1e6 * (1 - x ^ 2)
const bnd = TEQUILA.MXH(1.7, -0.05, 0.35, 1.8, 0.07, [0.04, -0.05, 0.02, 0.01], [0.64, 0.08, -0.09, 0.03])
psi(r,z) = -2.0 * (1.0 - (r - bnd.R0) ^ 2 - (z - bnd.Z0) ^ 2)
const Pbnd = 700.
const Fbnd = -3.5
const Ip_target = 6e5
const x = range(0, 1, 11)
const psi_fe = TEQUILA.FE(x, (x .^ 2) .- 2.0)

@testset "TEQUILA.jl" begin
    @test Shot(10, 12, bnd; dP_dψ=(Pp, :poloidal), F_dF_dψ=(FFp, :toroidal), Pbnd, Fbnd, Ip_target) isa Shot
    @test Shot(12, 10, bnd, psi ; P=(P, :toroidal), Jt=(J, :poloidal), Pbnd, Fbnd) isa Shot
    shot = Shot(11, 11, bnd, psi_fe; P=(P, :toroidal), Jt_R=(J, :toroidal), Pbnd, Fbnd, Ip_target)
    @test solve(shot, 5) isa Shot
    refill = solve(shot, 21; relax=0.5, tol=1e-3, debug=true, dP_dψ=(Pp, :toroidal), Jt=(J, :toroidal))
    @test refill isa Shot
    @test isapprox(shot.Ip_target, Ip(refill); rtol=1e-2)
end
