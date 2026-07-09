# VEQ direct-solve validation against the veqpy reference implementation.
#
# Reference data in veq_ref_case.jl was generated from the veqpy demo case
# (github.com/zhangtakeda/veqpy, demo.py topology) by evaluating the packed
# residual at x=0 and at a fixed pseudo-random x_ref, plus the converged
# solution vector. See docs/veq_formulation.md.

using TEQUILA
using Test
using LinearAlgebra

include(joinpath(@__DIR__, "veq_ref_case.jl"))

@testset "VEQ direct solve" begin
    top = VEQTopology(;
        h_count=3, v_count=0, kappa_count=6, psin_count=6, F_count=0,
        c_counts=Int[], s_counts=[3], Nr=16, Nt=16,
        route=:PF, coordinate=:psin, nodes=:uniform,
        ip_constraint=true, sample_count=51)
    boundary = VEQBoundary(; a=1.05 / 1.85, R0=1.05, Z0=0.0, B0=3.0, ka=2.2,
        s_offsets=[asin(0.5)])
    source = VEQSource(; heat_profile=HEAT, current_profile=CURR, Ip=3.0e6)
    kern = VEQKernel(top, boundary, source)

    @test kern.x_size == 18

    # packed residual matches veqpy at two states
    @test maximum(abs.(veq_residual(zeros(kern.x_size), kern) .- R0)) < 1e-7
    @test maximum(abs.(veq_residual(X_REF, kern) .- R_REF)) < 1e-7

    # veqpy's converged solution is (numerically) a root of our residual
    @test norm(veq_residual(X_SOLUTION, kern)) < 1e-6

    # cold-start Newton converges to the same solution
    x, ok, rn, it = veq_solve(kern)
    @test ok
    @test rn < 1e-9
    @test it <= 15
    @test maximum(abs.(x .- X_SOLUTION)) < 1e-6
end

@testset "veq_solve! vs Picard" begin
    Pp(x) = -1e5 * (1 - x^2)
    FFp(x) = 3.0 * (1 - x^2)
    bnd = TEQUILA.MXH(1.7, -0.05, 0.35, 1.8, 0.07, [0.04, -0.05, 0.02, 0.01], [0.64, 0.08, -0.09, 0.03])
    Pbnd, Fbnd = 700.0, -3.5

    shot_p = Shot(11, 11, bnd; dP_dψ=(Pp, :poloidal), F_dF_dψ=(FFp, :poloidal), Pbnd, Fbnd)
    picard = solve(shot_p, 30; tol=1e-10)
    _, _, Ψax_p = find_axis(picard)

    shot_v = Shot(11, 11, bnd; dP_dψ=(Pp, :poloidal), F_dF_dψ=(FFp, :poloidal), Pbnd, Fbnd)
    veq_solve!(shot_v; h_count=5, v_count=5, kappa_count=6, c0_count=5, psin_count=7,
        c_counts=[4, 4, 3, 3], s_counts=[4, 4, 3, 3], Nr=16, Nt=32)
    Rax_v, Zax_v, Ψax_v = find_axis(shot_v)

    @test abs(Ψax_v - Ψax_p) / abs(Ψax_p) < 0.01
    @test isapprox(Rax_v, 1.700, atol=0.01)
    @test isapprox(Zax_v, -0.013, atol=0.01)
    # interior surfaces track the Picard solution
    for k in (4, 7, 10)
        @test isapprox(shot_v.surfaces[3, k], picard.surfaces[3, k]; rtol=2e-2)  # ϵ
        @test isapprox(shot_v.surfaces[4, k], picard.surfaces[4, k]; rtol=2e-2)  # κ
    end
end

@testset "veq_solve! P/Jt/Ip route (FUSE inputs) vs Picard" begin
    # FUSE's ActorTEQUILA passes pressure and toroidal current density on the
    # rho_tor_norm grid with an Ip target; conversion to dP_dψ/FF' depends on
    # the evolving equilibrium, exercising the per-iteration writeback path.
    P_f(x) = 1.2e5 * (1 - x^2)^2 + 700.0
    Jt_f(x) = 1.1e6 * (1 - x^2)
    bnd = TEQUILA.MXH(1.7, -0.05, 0.35, 1.8, 0.07, [0.04, -0.05, 0.02, 0.01], [0.64, 0.08, -0.09, 0.03])
    Pbnd, Fbnd, Ip_target = 700.0, 3.4, 1.3e6

    shot_p = Shot(11, 11, bnd; P=(P_f, :toroidal), Jt=(Jt_f, :toroidal), Pbnd, Fbnd, Ip_target)
    picard = TEQUILA.solve(shot_p, 100; tol=1e-10)
    _, _, Ψax_p = find_axis(picard)

    shot_v = Shot(11, 11, bnd; P=(P_f, :toroidal), Jt=(Jt_f, :toroidal), Pbnd, Fbnd, Ip_target)
    veq_solve!(shot_v; h_count=5, v_count=5, kappa_count=6, c0_count=5, psin_count=7,
        c_counts=[4, 4, 3, 3], s_counts=[4, 4, 3, 3], Nr=16, Nt=32)
    _, _, Ψax_v = find_axis(shot_v)

    @test abs(Ψax_v - Ψax_p) / abs(Ψax_p) < 0.02
    @test isapprox(TEQUILA.Ip(shot_v), Ip_target; rtol=1e-6)
    for k in (4, 7, 10)
        @test isapprox(shot_v.surfaces[3, k], picard.surfaces[3, k]; rtol=2e-2)  # ϵ
        @test isapprox(shot_v.surfaces[4, k], picard.surfaces[4, k]; rtol=2e-2)  # κ
    end

    # warm restart from its own solution converges immediately and stays put
    Ψax_prev = Ψax_v
    veq_solve!(shot_v; h_count=5, v_count=5, kappa_count=6, c0_count=5, psin_count=7,
        c_counts=[4, 4, 3, 3], s_counts=[4, 4, 3, 3], Nr=16, Nt=32)
    _, _, Ψax_w = find_axis(shot_v)
    @test isapprox(Ψax_w, Ψax_prev; rtol=1e-4)
end
