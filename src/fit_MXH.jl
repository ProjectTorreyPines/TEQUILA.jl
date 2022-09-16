function coordinates!(pr, pz, c::Curve2)
    for (i, v) in enumerate(c.vertices)
        pr[i] = v[1]
        pz[i] = v[2]
    end
    return pr, pz
end

# function refit(shot::Shot, lvls::AbstractVector{<:Real})#; inner_optimizer::Optim.FirstOrderOptimizer=Optim.ConjugateGradient())

#     refill = deepcopy(shot)

#     R0 = shot.surfaces[1, end]
#     Z0 = shot.surfaces[2, end]
#     a = R0 * shot.surfaces[3, end]
#     b = a * shot.surfaces[4, end]

#     Nr = 201
#     Nz = 201
#     Rs = range(R0 - a, R0 + a, Nr)
#     Zs = range(Z0 - b, Z0 + b, Nz)
#     for (i, r) in enumerate(Rs)
#         for (j, z) in enumerate(Zs)
#             Ψcntr[j, i] = shot(r, z)
#         end
#     end

#     @views cntrs = contours(Rs, Zs, transpose(Ψcntr), lvls[2:end-1])

#     Nl = 0
#     for cl in levels(cntrs)
#         l = first(lines(cl))
#         length(l.vertices) > Nl && (Nl = length(l.vertices))
#     end
#     pr = zeros(Nl)
#     pz = similar(pr)
#     θ = similar(pr)
#     Δθᵣ = similar(pr)
#     dθ = similar(pr)
#     Fm = similar(pr)
#     S = refill.surfaces
#     #p = plot(aspect_ratio=:equal)
#     for (k, cl) in enumerate(levels(cntrs))
#         l = first(lines(cl))
#         Nl = length(l.vertices)
#         @views coordinates!(pr[1:Nl], pz[1:Nl], l) # coordinates of this line segment
#         #@views plot!(p, pr[1:Nl], pz[1:Nl],lw=3, color=:black)
#         @views fit_flattened!(S[:, k+1], pr[1:Nl], pz[1:Nl], θ[1:Nl], Δθᵣ[1:Nl], dθ[1:Nl], Fm[1:Nl])
#         #@views plot!(p, MXH(refill.surfaces[:, k+1]), lw=2, color=:red, ls=:dash)
#     end
#     #display(p)

#     # Extrapolate or set to zero on-axis
#     ρ2 = sqrt((lvls[2]-lvls[1])/(lvls[end]-lvls[1]))
#     ρ3 = sqrt((lvls[3]-lvls[1])/(lvls[end]-lvls[1]))
#     h = 1.0 / (ρ3 - ρ2)
#     S[1, 1] = h .* (S[1, 2] .* ρ3 .- S[1, 3] .* ρ2)
#     S[2, 1] = h .* (S[2, 2] .* ρ3 .- S[2, 3] .* ρ2)
#     S[3, 1] = 0.0
#     S[4, 1] = h .* (S[4, 2] .* ρ3 .- S[4, 3] .* ρ2)
#     S[5, 1] = h .* (S[5, 2] .* ρ3 .- S[5, 3] .* ρ2)
#     @views S[6:end, 1] .= 0.0

#     Fi, _, Fo, P = fft_prealloc(refill.M)
#     S_FE = surfaces_FE(refill.ρ, refill.surfaces)

#     function Ψ_ρθ(x, t)
#         R, Z = R_Z(S_FE..., x, t)
#         return shot(R, Z)
#     end
#     refill.C .= 0.0
#     compute_Cmatrix!(refill.C, refill.N, refill.M, refill.ρ, Ψ_ρθ, refill._Afac, Fi, Fo, P)

#     #C = compute_Cmatrix(N, M, ρ, Ψ_ρθ)
#     #refill.C .= compute_Cmatrix()

#     # S_FE = surfaces_FE(shot.ρ, refill.surfaces)

#     # function Ψ_ρθ(x, t)
#     #     return shot(R_Z(S_FE..., x, t)...)
#     # end
#     # #C = zeros(2 * shot.N, 2 * shot.M + 1)
#     # refill. C .= 0.0
#     # Fi, _, Fo, P = fft_prealloc(shot.M)
#     # compute_Cmatrix!(refill.C, shot.N, shot.M, shot.ρ, Ψ_ρθ, shot._Afac, Fi, Fo, P)
#     #C = compute_Cmatrix(shot.N, shot.M, shot.ρ, Ψ_ρθ)
#     #refill = Shot(shot.N, shot.M, shot.ρ, refill.surfaces, shot)
#     #refill.C .= C
#     return refill
# end

function refit(shot::Shot, lvls::AbstractVector{<:Real})#; inner_optimizer::Optim.FirstOrderOptimizer=Optim.ConjugateGradient())

    R0 = shot.surfaces[1, end]
    Z0 = shot.surfaces[2, end]
    a = R0 * shot.surfaces[3, end]
    b = a * shot.surfaces[4, end]

    Rs = range(R0 - a, R0 + a, Ncntr)
    Zs = range(Z0 - b, Z0 + b, Ncntr)
    for (i, r) in enumerate(Rs)
        for (j, z) in enumerate(Zs)
            Ψcntr[j, i] = shot(r, z)
        end
    end

    @views cntrs = contours(Rs, Zs, transpose(Ψcntr), lvls[2:end-1])

    #L = length(shot._cx)
    #mxh = MXH(0.0, 0.0, 0.0, 0.0, 0.0, zeros(L), zeros(L))
    Nl = 0
    for cl in levels(cntrs)
        l = first(lines(cl))
        length(l.vertices) > Nl && (Nl = length(l.vertices))
    end
    pr = zeros(Nl)
    pz = similar(pr)
    θ = similar(pr)
    Δθᵣ = similar(pr)
    dθ = similar(pr)
    Fm = similar(pr)

    surfaces = deepcopy(shot.surfaces)
    @views for (k, cl) in enumerate(levels(cntrs))
        l = first(lines(cl))
        Nl = length(l.vertices)
        @views coordinates!(pr[1:Nl], pz[1:Nl], l)
        #pr, pz = coordinates(lines(cl)[1]) # coordinates of this line segment
        #Nl = length(pr)
        #θ = similar(pr)
        #Δθᵣ = similar(pr)
        #dθ = similar(pr)
        #Fm = similar(pr)
        #MXH!(mxh, pr, pz; θ, Δθᵣ, dθ, Fm)
        #@views flat_coeffs!(surfaces[:, k+1], mxh)
        @views fit_flattened!(surfaces[:, k+1], pr[1:Nl], pz[1:Nl], θ[1:Nl], Δθᵣ[1:Nl], dθ[1:Nl], Fm[1:Nl])
    end

    # Allocate initial guess and bounds
    #@views x0 = zero(shot.surfaces[:,1])
    #lower = zero(x0)
    #upper = zero(x0)

    #loop_surfaces!(shot_refit, lvls, x0, lower, upper; inner_optimizer)

    # Extrapolate or set to zero on-axis
    ρ2 = sqrt((lvls[2]-lvls[1])/(lvls[end]-lvls[1]))
    ρ3 = sqrt((lvls[3]-lvls[1])/(lvls[end]-lvls[1]))
    h = 1.0 / (ρ3 - ρ2)
    surfaces[1, 1] = h .* (surfaces[1, 2] .* ρ3 .- surfaces[1, 3] .* ρ2)
    surfaces[2, 1] = h .* (surfaces[2, 2] .* ρ3 .- surfaces[2, 3] .* ρ2)
    surfaces[3, 1] = 0.0
    surfaces[4, 1] = h .* (surfaces[4, 2] .* ρ3 .- surfaces[4, 3] .* ρ2)
    surfaces[5, 1] = h .* (surfaces[5, 2] .* ρ3 .- surfaces[5, 3] .* ρ2)
    @views surfaces[6:end, 1] .= 0.0

    #remap_shot!(shot, surfaces)
    shot_refit = Shot(shot.N, shot.M, shot.ρ, surfaces, shot)

    return shot_refit
end