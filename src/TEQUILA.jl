__precompile__()
module TEQUILA

using FiniteElementHermite
using MillerExtendedHarmonic
import NLsolve
using Plots
import QuadGK: quadgk
import FFTW: fft
#using Trapz
#using BandedMatrices
#import ForwardDiff

const μ₀ = 4e-7*π

include("initialize.jl")
export Ψmiller, first_shot

include("shot.jl")
export TEQUILAshot, plot_shot

include("FE_Fourier.jl")

include("surfaces.jl")
export concentric_surface, surfaces_FE, R_Z, ρ_θ

end
