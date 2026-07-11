using PrecompileTools: @compile_workload

@compile_workload begin
    bnd = MXH(1.7, -0.05, 0.35, 1.8, 0.07, [0.04, -0.05, 0.02, 0.01], [0.64, 0.08, -0.09, 0.03])
    Pp(x) = -1e5 * (1 - x^2)
    FFp(x) = 3.0 * (1 - x^2)
    x = range(0, 1, 11)
    psi_fe = FE(x, (x .^ 2) .- 2.0)
    shot = Shot(11, 11, bnd, psi_fe; dP_dψ=(Pp, :toroidal), F_dF_dψ=(FFp, :toroidal), Pbnd=700.0, Fbnd=-3.5, Ip_target=6e5)
    solve(shot, 3)
end
