# VEQ-style direct solve: formulation notes

Reference: VEQ (Veloce EQuilibrium), arXiv:2606.11821, and its Python
implementation VEQPy (`veqpy`, BSD-3). These notes are reverse-engineered from
the `veqpy` Numba kernel (`veqpy/kernels/numba_kernel/*.py`) to guide the
TEQUILA implementation in `src/veq.jl`. Section references below name the
veqpy source files that define each piece.

## Idea

Fixed-boundary Grad-Shafranov solve where the unknowns are **radial profiles
of MXH shape coefficients** (not 2D flux values). The GS strong-form residual
is evaluated on a small quadrature grid and projected variationally — each
unknown coefficient gets the residual weighted by its own *virtual
displacement* — giving a square nonlinear system of dimension ~10–130 solved
with Powell hybrid / Levenberg-Marquardt. Solve times: 0.25–3 ms (C++), 2–14 ms
(Numba).

## Radial representation (`profile_stage.py`, `model/grid.py`)

Coordinates on ρ ∈ [0,1] (ρ = normalized minor-radius-like flux label):

- Chebyshev argument: `x = 2ρ² − 1` (even parity in ρ automatic)
- Edge envelope:      `y = 1 − ρ²`   (test/trial envelope; kills boundary variation → fixed boundary)

Every profile has the form

```
p(ρ) = scale · ρ^power · [ offset + y^env_power · Σ_l c_l T_l(x) ]^amp_power
```

with analytic ρ- and ρρ-derivatives assembled by product/chain rule.
`offset` is the **boundary value** (envelope vanishes at ρ=1), taken from the
boundary MXH fit. Per-family parameters (`packed_layout.py`):

| family | meaning                        | offset       | power        | env | amp |
|--------|--------------------------------|--------------|--------------|-----|-----|
| h      | R0 shift / a                   | 0            | 0            | 1   | 1   |
| v      | Z0 shift / a                   | 0            | 0            | 1   | 1   |
| k      | elongation κ(ρ)                | ka (bnd κ)   | 0            | 1   | 1   |
| c0     | θb tilt                        | c0 bnd       | 0            | 1   | 1   |
| c_m    | MXH cos coefficient            | c_m bnd      | K_m          | 1   | 1   |
| s_m    | MXH sin coefficient            | s_m bnd      | K_m          | 1   | 1   |
| psin   | normalized flux ψn(ρ)          | 1            | 2            | 1   | 1   |
| F      | (F/R0B0)² amplitude            | 1            | 0            | 1   | 0.5 |

- `K_m = min(m, K_max)` axis-regularity power for Fourier order m (`grid.py:_build_K_values`).
- F profile: coefficients parameterize normalized F², evaluator applies sqrt;
  `scale = R0·B0` restores units. F(1) = R0·B0 = Fbnd.
- psin: ψn(ρ) = ρ²·[1 + y·Σ…], so ψn(0)=0, ψn(1)=1, ψn ~ ρ² at axis.

## Geometry (`geometry_stage.py`)

With boundary scalars `R0, Z0, a` (geometric center/minor radius) and the
shape profiles above:

```
θb(ρ,θ) = θ + c0(ρ) + Σ_m [ c_m(ρ)·cos(mθ) + s_m(ρ)·sin(mθ) ]
R(ρ,θ)  = R0 + a·( h(ρ) + ρ·cos θb )
Z(ρ,θ)  = Z0 + a·( v(ρ) − ρ·k(ρ)·sin θ )
```

(veqpy folds Z0 into v. CONFIRMED: this is the *same* convention as
MillerExtendedHarmonic — `Z_MXH = Z0 − κ a sin θ`, `R_MXH = R0 + a cos θr` —
so Shot boundary surfaces map to VEQBoundary with no sign flips.)

## TEQUILA interop (src/veq_shot.jl)

- TEQUILA ρ grid: ρ = sqrt(normalized poloidal flux), Ψbnd = 0, surface k at
  Ψ = Ψaxis(1−ρ_k²), i.e. ψn = ρ_k².
- **Flux convention**: TEQUILA Ψ is total flux (Wb); VEQ's GS residual is
  per-radian, so α2 = (ψ_s/2π)² with ψ_s = dΨ_TEQUILA/dψn = −Ψaxis.
  Confirmed numerically: the (2π)² factor reproduces Picard's Ψaxis to <0.5%.
- PF no-constraint inputs in physical units: heat = dP/dψn = ψ_s·dP/dΨ,
  curr = d(F²/2)/dψn = ψ_s·F dF/dΨ. Both scale linearly with ψ_s and leave
  the solved shape invariant ⇒ α2 is exactly linear in ψ_s ⇒ the outer
  fixed point ψ_s ← 4π²·α2/ψ_s converges in ~2 iterations.
- Writeback: evaluate MXH surfaces analytically at ψn = ρ_k² (invert the
  smooth ψn(ρ_veq) profile), plus the two edge sub-surfaces (δ_frac_2/3),
  then `update_shot!(shot, surfaces, −ψ_s, flat_δ2, flat_δ3)` rebuilds all
  FE fields/quadrature/FSAs and sets C to the exact flux-function form.
- Validation (test_veq.jl): Ψaxis matches Picard to 0.4% at modest counts,
  converging 0.46% → 0.11% as the representation is enriched.

First and second derivatives (R_ρ, R_θ, R_ρρ, R_ρθ, R_θθ, same for Z) are
analytic. Derived per-point fields (`gtt` = covariant g_θθ, `grt` = g_ρθ):

```
J        = R_θ Z_ρ − R_ρ Z_θ          (clamped at 1e-6 to avoid fold NaNs)
gtt      = R_θ² + Z_θ²
grt      = R_ρ R_θ + Z_ρ Z_θ
gttdivJR   = gtt/(J R)
gttdivJR_r = ∂ρ(gtt/(J R))
grtdivJR_t = ∂θ(grt/(J R))
JdivR    = J/R
```

FSA radial moments (θ-sums over the uniform θ grid, `mean_scale = 1/Nt`,
`theta_scale = 2π/Nt`):

```
S'(ρ)  = ∮ J dθ                     (2π/Nt · Σ_j J)
V'(ρ)  = 2π ∮ J R dθ
Kn(ρ)  = ⟨gtt/(J R)⟩ = (1/Nt) Σ_j gtt/(J R)
Kn_r   = (1/Nt) Σ_j ∂ρ(gtt/(J R))
Ln'(ρ) = ⟨J/R⟩
```

## Strong-form residual (`numba_residual.py::update_residual_compact`)

For a flux function ψ = ψ(ρ), using normalized ψn and normalized source
profiles Pn′(ψn), FFn′(ψn):

```
G1n(ρ,θ) = (J/R)·( FFn′ + R²·Pn′ )        # source part
G2n(ρ,θ) = gttdivJR·ψn″ + (gttdivJR_r − grtdivJR_t)·ψn′   # = (J/R)·Δ*ψn
G = α1·G1n + α2·G2n
```

G = 0 pointwise ⇔ GS equation; α1, α2 are route-dependent global scalings
(physical units / Ip / beta constraints) returned by the source stage.

## Variational projection (`numba_residual.py::run_residual_blocks_packed_precomputed`)

Quadrature: Legendre nodes/weights `(ρ_i, w_i)` on (0,1) radially; uniform θ
with trapezoid weight `2π/Nt`. Precompute `ψn_R = −Z_θ/J·ψn′`,
`ψn_Z = R_θ/J·ψn′` (∇ψ components) and cache `G·ψn_R`, `G·ψn_Z`,
`G·ψn_R·sin θb`.

Residual entry for coefficient l of each block (T_l = T_l(x(ρ)), all radial
sums weighted by `w_i`, θ-sums by `2π/Nt`):

| block | equation (θ-collapse, then radial projection vs T_l) |
|-------|------------------------------------------------------|
| h     |  +a · ΣΣ (G·ψn_R) · y · T_l |
| v     |  +a · ΣΣ (G·ψn_Z) · y · T_l |
| k     |  −a · ΣΣ (G·ψn_Z)·sinθ · ρ · y · T_l |
| c0    |  −a · ΣΣ (G·ψn_R·sinθb) · ρ · y · T_l |
| c_m   |  −a · ΣΣ (G·ψn_R·sinθb)·cos(mθ) · ρ^(K_m+1) · y · T_l |
| s_m   |  −a · ΣΣ (G·ψn_R·sinθb)·sin(mθ) · ρ^(K_m+1) · y · T_l |
| psin  |      ΣΣ G · ρ² · y · T_l |
| F     |  (R0·B0)² · ΣΣ G · y² · T_l |

These are exactly ∫∫ G · (∂geometry/∂coeff)-induced virtual flux
displacements: e.g. ∂R/∂h_l = a·y·T_l ⇒ δψ ~ ψn_R·a·y·T_l. Square system by
construction: one row per active coefficient.

## Source routes (`numba_source.py`, `abi/source_semantics.py`)

All routes fill one contract: `psin, psin_r, psin_rr` (radial arrays),
`Pn′(ψn(ρ_i))`, `FFn′(ψn(ρ_i))` (arrays on the ρ grid), and scalars
`(α1, α2)`. Routes: PF (pressure + current-ish pair), PP, PI, PJ1, PJ2, PQ.
`coordinate ∈ {rho, psin}` says which axis the user profiles are sampled on;
`nodes ∈ {uniform, grid}` whether they're on a uniform axis (needs
interpolation to quadrature nodes) or already on the grid.

Active family rule (`types.py:_source_active_family`): route PJ2 ⇒ "F";
`coordinate="psin" & nodes="uniform"` ⇒ "psin" (psin block REQUIRED active in
x); else "none". Parameterization "sqrt_psin" only for PP/psin/uniform.

### PF, coordinate=psin, nodes=uniform, Ip constraint (demo case) — CONFIRMED NUMERICALLY

Residual root fields (drive G and source sampling) come from the **x psin
block** (profile stage):
1. `psin_r(ρ_i)` = profile-stage ψn′ from x, then `_regularize_psin_r`:
   linear-in-ρ² extrapolation of ψn′/ρ from anchors i=n_fix, n_fix+1 back to
   the first n_fix samples ("fix_rho" threshold sets n_fix; =2 for Nr=16),
   floored at 1e-10.
2. `psin(ρ_i)` = spectral cumulative integral (`full_integration` with the
   Legendre "accumulator" matrix) of regularized psin_r, then normalized
   exactly to psin[0]=0, psin[-1]=1. This is also `psin_query`.
3. `psin_rr` = spectral differentiation of regularized psin_r.
4. Materialized inputs: heat/current user samples (uniform ψn axis,
   `sample_count` pts) → cubic-spline coefficients → evaluated at
   `psin_query`. **μ0 scaling**: `Pn_psin = μ0·heat(ψn)` (pressure-like
   inputs scaled by μ0 at materialization); `FFn_psin = curr(ψn)` with
   `_regularize_ffn_psin` = even-in-ρ (linear in ρ²) axis fix from samples
   1,2.
5. FSA-derived *target* ψn (written to `sw.target_root_fields`, NOT used in
   G): integrand `I_i = curr_i·Ln′_i + V′_i·heat_i/(4π²)`;
   `t_r = −∫I/Kn` (cumulative), sign-fixed positive, regularized,
   `c2 = ∫ t_r w` (before normalizing t_r to unit integral).
6. Alphas with Ip: `G1n_integral = Σ w_i·(2π·Ln′_i·FFn_i + V′_i·Pn_i/2π)`;
   `α1 = −Ip/G1n_integral`; `α2 = c2·α1`. (No-constraint and beta variants
   in `numba_source.py:1290`.)

Reference dumps for cross-validation (demo case, x_size=18, layout
"degree"-interleaved: h idx [0,4,8], k [1,5,9,12,14,16], s1 [2,6,10],
psin [3,7,11,13,15,17]) live in the Claude scratchpad `veq_ref_case.npz`
(x_ref/r_ref/alphas/x_solution) — regenerate with `dump_veq_case.py` there
against the veqpy checkout `.venv`.

## Solver (`solver.py`)

- x0 = 0 (cold start; offsets carry boundary values so x=0 ⇒ concentric-ish
  surfaces scaled by ρ) with two small heuristic tweaks (initialize.py: h0 and
  first-coefficient estimates).
- scipy `root(method="hybr")` (Powell / MINPACK hybrd) or `least_squares(lm)`.
  Tolerances from `max_residual`; optional residual normalization ("balanced",
  "safe") and x-scaling transform for hybr.
- Acceptance: ‖raw residual‖ ≤ max(max_residual·factor, floor).
- Warm continuation: reuse previous x between solves on the same handle.

## Julia implementation plan (src/veq.jl)

- Same formulation, same quadrature. ForwardDiff exact Jacobian + dense
  Newton/LM (system is tiny), PreallocationTools for Dual-compatible buffers.
- Unknowns exactly as VEQ (Chebyshev in x = 2ρ²−1) — NOT the Shot Hermite-FE
  basis — for direct numerical cross-validation against veqpy. Conversion
  Shot ↔ VEQ state: evaluate shape profiles at knots (values + derivatives)
  and FE-fit; ψ from ψn profile × edge flux.
- Boundary input from `MillerExtendedHarmonic.MXH` (mind θ-sign convention
  above) or from a Shot's boundary surface.
- Entry point: `veq_solve!(shot; route=:PF, ...)` and a lower-level
  `VEQKernel` mirroring topology/case split for scans.

## Validation plan

1. Unit: Julia profile/geometry/moment fields vs veqpy dumps on identical x
   (veqpy checkout has `.venv`; dump with small driver script).
2. Residual: identical packed residual for identical x on the demo.py case
   (h=3, k=6, psin=6, s1=3, Nr=Nt=16, PF/psin/uniform, Ip=3e6).
3. End-to-end: veq_solve! vs TEQUILA Picard on the same P/FF' inputs
   (SOLOVEV/CHEASE/EFIT geqdsks in veqpy/data), compare ψ(ρ), surfaces, q.
4. Benchmark: target ≤ C++ backend times (0.25–3 ms).
