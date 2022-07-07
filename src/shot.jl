mutable struct Shot # <: Abst]practEquilibrium (eventually)
    N :: Integer
    M :: Integer
    ρ :: AbstractVector{<:Real}
    surfaces :: AbstractVector{<:MXH}
    C :: AbstractMatrix{<:Real}
end

function plot_shot(shot::Shot, axes=:rz, points=101)

    function g(x, t) 
        return sum([shot.C[2i-1,2m+1] * νo(x,i,shot.ρ) * cos(m*t) +
                   shot.C[2i  ,2m+1] * νe(x,i,shot.ρ) * cos(m*t) +
                   shot.C[2i-1,2m+2] * νo(x,i,shot.ρ) * sin(m*t) +
                   shot.C[2i  ,2m+2] * νe(x,i,shot.ρ) * sin(m*t)
                   for i in 1:shot.N, m in 0:shot.M])
    end

    if axes == :ρθ
        xs = range(0, 1, points)
        ys = range(0, 2π, points)
        G = [g(x, y) for y in ys, x in xs]
        return heatmap(xs, ys, G)
    elseif axes == :rz

        boundary = shot.surfaces[end]
        a = boundary.R0 * boundary.ϵ
        Rmin = boundary.R0 - a
        Rmax = boundary.R0 + a

        Zmin = boundary.Z0 - a * boundary.κ
        Zmax = boundary.Z0 + a * boundary.κ
        
        xs = range(Rmin, Rmax, points)
        ys = range(Zmin, Zmax, points)

        S_FE = surfaces_FE(shot)

        G = zeros(points, points)
        for (i,x) in enumerate(xs)
            for (j,y) in enumerate(ys)
                r, z = ρ_θ(S_FE..., x, y)
                if r == NaN
                    G[j,i] = NaN 
                else
                    G[j,i] = g(r,z)
                end
            end
        end
        return heatmap(xs, ys, G, aspect_ratio=:equal, clim=(0,1))
    end
end

#function (S::Shot)(R, Z)



