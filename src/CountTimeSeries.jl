module CountTimeSeries

using Optim, Distributions, LinearAlgebra, Plots, Calculus, Roots, Random
using Test, StatsBase, Distributed

include("ModelTypes.jl")
include("parameter.jl")
include("MLEControl.jl")
include("results.jl")

include("AIC.jl")
include("BIC.jl")
include("convert.jl")
include("convolution.jl")
include("fit.jl")
include("GetProbFromP.jl")
include("HQIC.jl")
include("LinPred.jl")
include("ll.jl")
include("MLESettings.jl")
include("Model.jl")
include("par2theta.jl")
include("parametercheck.jl")
include("pit.jl")
include("predict.jl")
include("QPois.jl")
include("show.jl")
include("simulate.jl")
include("theta2par.jl")
include("thinning.jl")

# Functions, types and structs with documentation
export AIC, BIC, fit, HQIC, ll, MLEControl, MLESettings, Model, CountModel
export INGARCH, INARMA, IIDModel, INARCHModel, INGARCHModel, INARMAModel
export INARModel, INMAModel, par2θ, parameter, pit, predict, QPois
export INGARCHresults, INARMAresults, simulate, θ2par, ∘

end
