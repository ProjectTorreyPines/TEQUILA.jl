# VEQ direct-solve validation against the veqpy reference implementation.
#
# Reference data in veq_ref_case.jl was generated from the veqpy demo case
# (github.com/zhangtakeda/veqpy, demo.py topology) by evaluating the packed
# residual at x=0 and at a fixed pseudo-random x_ref, plus the converged
# solution vector. See docs/veq_formulation.md.

using TEQUILA
using Test
using LinearAlgebra

# Reference data from veqpy demo case (dump_veq_case.py)
const X_REF = [-0.028476500729092625, 0.02527456916258221, -0.017413234759181714, -0.005183464698687952, -0.0015068661402104194, -0.014817693041712182, -0.02735585403565887, 0.012977856043860798, 0.0072211622610979, -0.0390572612602438, 0.046948193087577035, 0.01936993811503847, -0.015187743608490132, 0.018043965484245035, -0.009339063466411005, -0.0012137903747405595, 0.015776886890384017, -0.02513336266279353]
const R_REF = [-0.4989577438630739, 0.09991226589565272, 0.03299908851094133, 0.2769547342683474, -0.09466516559262549, 0.05163145940915006, 0.010715189916327963, 0.1269511833713899, 0.31013922224005325, -0.07118209508919163, -0.007027843118720536, -0.19605878454362685, -0.1126676324663064, -0.26791034030456684, 0.005257175634340978, 0.040604023335893684, 0.1028886562836137, 0.25509862807949474]
const ALPHA_REF = [6.435733181993649, 0.23762962286788328]
const R0 = [-0.3672906243964035, 0.08082898496085911, 0.029351809936649865, 0.23784496708590314, -0.08527207963154187, 0.04542855345169576, 0.007432973699126383, 0.12159840249136648, 0.1952638805604906, -0.03511305183648078, -0.016434081073438453, -0.10533473890844339, -0.05498276616796438, -0.1466432925965151, -0.010210064494201457, -0.027371133827348605, 0.006265668063701097, 0.016099122028397715]
const ALPHA0 = [6.514270192929262, 0.24225579564650504]
const X_SOLUTION = [0.07146980888382737, -0.5789808694576329, -0.4209226935191592, 0.18118220393036402, 0.009853215499737649, -0.4044530378398727, -0.2317387682308575, 0.04891735939437488, -0.0006309744175887253, -0.13584593701148137, -0.13538592999266483, 0.010043050547339456, -0.036425936907335796, 0.004058994840433141, -0.0017151082914709783, 0.0026276217226210157, 0.00011403191136843831, 0.00026826420330055303]
const HEAT = [-0.9296172467538416, -0.9289540176371517, -0.9282210361053653, -0.927410966232948, -0.9265157005683434, -0.9255262789918709, -0.9244327990398366, -0.9232243167973497, -0.9218887373679424, -0.9204126938237812, -0.9187814134249611, -0.9169785697689582, -0.9149861193905066, -0.9127841211765332, -0.9103505367887953, -0.9076610100967843, -0.9046886234133856, -0.901403628093618, -0.8977731468001962, -0.8937608444560887, -0.8893265645908547, -0.8844259274411892, -0.8790098857833379, -0.8730242340519946, -0.8664090658327849, -0.8590981742987359, -0.8510183895900996, -0.8420888465058022, -0.8322201751773333, -0.8213136066250626, -0.8092599842450945, -0.7959386713332881, -0.7812163437125739, -0.7649456553797788, -0.7469637638173009, -0.7270907002104605, -0.705127568259113, -0.6808545535566255, -0.6540287236134139, -0.6243815965069333, -0.591616453824359, -0.5554053710049784, -0.5153859353609705, -0.4711576189294132, -0.4222777698538083, -0.3682571821755234, -0.30855519969612666, -0.24257430490843607, -0.1696541388404715, -0.08906489196090966, 0.0]
const CURR = [-0.3395336409686904, -0.33865929484102064, -0.3377249212727058, -0.33672639913423397, -0.33565932436368295, -0.33451899054225254, -0.33330036813622704, -0.33199808231381345, -0.33060638923901536, -0.32911915073798403, -0.32752980722611036, -0.32583134877645104, -0.32401628420188466, -0.3220766080146298, -0.3200037651173999, -0.31778861307046097, -0.31542138176816814, -0.312891630347132, -0.3101882011359537, -0.3072991704434206, -0.30421179596811115, -0.3009124605974526, -0.29738661234835495, -0.29361870018452313, -0.2895921054273651, -0.28528906845797763, -0.28069061038692417, -0.2757764493463215, -0.27052491103503595, -0.26491283312244135, -0.25891546308910296, -0.25250634905380676, -0.2456572231054175, -0.23833787662499076, -0.23051602704823781, -0.22215717548068875, -0.2132244545375532, -0.2036784657371647, -0.19347710573081978, -0.182575380602584, -0.17092520742001918, -0.1584752021605566, -0.14517045307814655, -0.13095227851060168, -0.11575796805942255, -0.09952050600055873, -0.08216827570618693, -0.06362474377383355, -0.04380812246966976, -0.022631008997155415, 0.0]

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
