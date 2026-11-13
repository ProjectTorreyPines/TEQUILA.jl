using TEQUILA
using Test
import ForwardDiff
import MXHEquilibrium

# Forward-mode derivatives through veq_solve!: values must equal the Float64
# solve and partials must match central finite differences of Float64 solves.
@testset "veq_solve! with ForwardDiff duals" begin
    Tag = typeof(ForwardDiff.Tag(veq_solve!, Float64))
    kw = (h_count=5, v_count=5, kappa_count=8, c0_count=5, psin_count=13,
        c_counts=[4, 4, 3, 3], s_counts=[4, 4, 3, 3], Nr=26, Nt=32, outer_tol=1e-12, tol=1e-11)
    P_f(x) = 1.2e5 * (1 - x^2)^2 + 700.0
    Jt_f(x) = 1.1e6 * (1 - x^2)
    bnd = TEQUILA.MXH(1.7, -0.05, 0.35, 1.8, 0.07, [0.04, -0.05, 0.02, 0.01], [0.64, 0.08, -0.09, 0.03])
    ρg = collect(range(0, 1, 101))  # FUSE-like dense profile grid (P'(0) ≈ 0)
    # p = (pressure scale, Jt scale, Fbnd, Ip_target), the FUSE input route
    mkshot(p) = Shot(11, 11, bnd; P=(TEQUILA.FE(ρg, p[1] .* P_f.(ρg)), :toroidal),
        Jt=(TEQUILA.FE(ρg, p[2] .* Jt_f.(ρg)), :toroidal), Pbnd=700.0 * p[1], Fbnd=p[3], Ip_target=p[4])
    pts = ((1.9, 0.1), (2.25, 0.0), (2.31, 0.0), (2.4, 0.0))  # last two outside the boundary (ρ ≤ 1.05 and beyond)
    outputs(s) = [s.C[2, 1]; s.surfaces[3, 6]; s.surfaces[4, 6]; s.R0fe(0.0); s.Vp(1.0); s.ρtor(0.5); s.F(0.0);
        [s(R, Z; extrapolate=true) for (R, Z) in pts]; MXHEquilibrium.psi_gradient(s, pts[1]...)[1]]

    p0 = [1.0, 1.0, 3.4, 1.3e6]
    yv = outputs(veq_solve!(mkshot(p0); kw...))
    pd = [ForwardDiff.Dual{Tag}(p0[i], ntuple(j -> Float64(i == j), 4)...) for i in 1:4]
    sd = veq_solve!(mkshot(pd); kw...)
    @test eltype(sd.surfaces) <: ForwardDiff.Dual
    yd = outputs(sd)
    @test all(y -> all(isfinite, ForwardDiff.partials(y)), yd)
    @test ForwardDiff.value.(yd) ≈ yv rtol = 1e-8

    # the Jt amplitude is fixed by the Ip constraint: its partials vanish
    @test all(abs(ForwardDiff.partials(yd[k], 2)) <= 1e-8 * abs(ForwardDiff.partials(yd[k], 1)) for k in eachindex(yd))

    # central differences: the Float64 solve itself is noisy at ~1e-8 in ψ_s,
    # so steps are 1 % and tolerances reflect O(h²) plus that noise
    for (i, rtol) in ((1, 2e-3), (3, 3e-2), (4, 2e-3))
        h = 1e-2 * p0[i]
        pp = copy(p0); pp[i] += h
        pm = copy(p0); pm[i] -= h
        fd = (outputs(veq_solve!(mkshot(pp); kw...)) .- outputs(veq_solve!(mkshot(pm); kw...))) ./ (2h)
        ad = [ForwardDiff.partials(y, i) for y in yd]
        scale = maximum(abs, fd)
        for k in eachindex(fd)
            @test isapprox(ad[k], fd[k]; rtol, atol=rtol * 1e-2 * scale)
        end
    end

    # Picard does not propagate duals
    @test_throws ArgumentError TEQUILA.solve(mkshot(pd), 2)
end
