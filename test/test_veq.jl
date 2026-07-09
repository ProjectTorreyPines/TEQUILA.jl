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
