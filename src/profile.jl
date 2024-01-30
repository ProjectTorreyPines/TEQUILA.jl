(prof::Profile)(x) = prof.fe(x)

deriv(y::Function, x) = ForwardDiff.derivative(y, x)
deriv(y::FE_rep, x) = D(y, x)
deriv(y::Profile, x) = D(y.fe, x)

make_profile(Y::Nothing, profile_grid::Symbol, ρtor) = nothing

function make_profile(Y::FE_rep, profile_grid::Symbol, ρtor)
    prof = Profile(deepcopy(Y), deepcopy(Y), profile_grid)
    update_profile!(prof, ρtor)
    return prof
end

function make_profile(Y::Function, profile_grid::Symbol, ρtor)
    N = length(ρtor.x) ^ 2 # lots of resolution
    x = range(0, 1, N)
    coeffs = zeros(2N) # this'll get defined in update_profile!
    fe = FE_rep(x, coeffs)
    prof = Profile(fe, Y, profile_grid)
    update_profile!(prof, ρtor; force=true)
    return prof
end

function make_profile(prof::Profile, profile_grid::Symbol, ρtor)
    if prof.grid !== profile_grid
        @warn "Profile given with grid $(prof.grid), but specified $(profile_grid). Will not update"
    end
    update_profile!(prof, ρtor)
    return prof
end

function update_profile!(prof::Profile, ρtor; force::Bool=false)
    fe = prof.fe
    if prof.grid === :toroidal
        fe.coeffs[1:2:end] .= deriv.(Ref(prof.orig), ρtor.(fe.x)) .* deriv.(Ref(ρtor), fe.x)
        fe.coeffs[2:2:end] .= prof.orig.(ρtor.(fe.x))
    elseif force
        fe.coeffs[1:2:end] .= deriv.(Ref(prof.orig), fe.x)
        fe.coeffs[2:2:end] .= prof.orig.(fe.x)
    end
    return prof
end

function update_profiles!(shot::Shot)
    profs = (shot.dP_dψ, shot.P, shot.F_dF_dψ, shot.Jt_R, shot.Jt)
    for prof in profs
        (prof !== nothing) && update_profile!(prof, shot.ρtor)
    end
    return shot
end