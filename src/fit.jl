# Wrapper for fitting Count Data Models

"""
    fit(y, model, MLEControl, printResults = true, initiate = "first")
Function to fit model to count time series.

* `y`: Time series
* `model`: Model specification
* `MLEControl`: Estimation settings
* `printResults`: Print results to console?
* `initiate`: How is conditional mean recursion initiated? "first", "marginal" or "intercept"

If MLEControl is not put in, default settings are used. Caution: This might lead to
poor initial values.
"""

# Function if MLEControl is given
import StatsBase.fit
# New functions here for restricted estimation
# Should include the case of no restrictions, old functions in comments below
function fit(y::Vector{Int64},
    model::T1,
    MLEControl::MLEControl;
    printResults::Bool=true,
    initiate::String="first",
    restr::Vector{Pair{String, T2}} = Vector{Pair{String, Float64}}(undef, 0))::Results where {T1<:INGARCH, T2 <: Real}
    T = length(y)
    nR = length(restr)

    if typeof(model) == INGARCHModel
        q = length(model.pastMean)
        Q = maximum(model.pastMean)
    else
        q = 0
        Q = 0
    end

    if typeof(model) != IIDModel
        p = length(model.pastObs)
        P = maximum(model.pastObs)
    else
        p = 0
        P = 0
    end

    r = length(model.external)
    M = maximum([P, Q])

    logl = (model.link == "Log")
    lin = !logl

    nb = model.distr == "NegativeBinomial"

    zi = model.zi

    nObs = T - M
    nPar = 1 + p + q + r + nb + zi - nR

    optimFun = BFGS()
    if MLEControl.optimizer == "LBFGS"
        optimFun = LBFGS()
    end

    if MLEControl.optimizer == "NelderMead"
        optimFun = NelderMead()
    end

    initVec = par2θ(MLEControl.init, model, restr)
    initVec = removeRestrictions(initVec, model, restr)
    
    if ll(y, model, initVec, restr, initiate=initiate)[1] == -Inf
        error("Infinite likelihood for initial parameters. Provide other values.")
    end

    optRaw = optimize(vars -> -ll(y, model, vars, restr, initiate=initiate)[1],
        initVec,
        optimFun,
        Optim.Options(iterations=MLEControl.maxEval))
    cvg = Optim.converged(optRaw)
    estsVec = optRaw.minimizer
    ests = θ2par(estsVec, model, restr)
    λ = LinPred(y, model, ests)

    temp = ll(y, model, ests, initiate=initiate)
    LLmax = temp[1]
    LLs = temp[2]

    resi = y .- λ

    se = zeros(nPar)
    CI = zeros(2, nPar)

    ciProb = false
    if nb
        if ests.ϕ[1] > 10000
            ciProb = true
            println("No confidence intervals computed due to large estimate of ϕ.")
        end
    end

    if MLEControl.ci & !ciProb
        H = Calculus.hessian(vars -> ll(y, model, vars, restr, initiate=initiate)[1], estsVec)
        if any(isnan.(H)) | any(.!isfinite.(H))
            ciProb = true
            println("No confidence intervals computed due to Inf/NaN in Hessian.")
        end
    end

    if MLEControl.ci & !ciProb
        Σ = -inv(H)
        se = sqrt.(diag(Σ))
        quan = quantile(Normal(), 0.975)
        for i = 1:nPar
            CI[:, i] = estsVec[i] .+ [-1, 1] .* se[i] .* quan
        end
    else
        Σ = zeros(nPar, nPar)
    end

    res = INGARCHresults(y, estsVec, ests, λ, resi, LLmax, LLs, nPar, nObs, se, CI, Σ, model, cvg, MLEControl)

    if printResults
        show(res, restr)
    end

    return res
end

function fit(y::Vector{Int64},
    model::T1;
    printResults::Bool=true,
    initiate::String="first",
    restr::Vector{Pair{String,T2}} = Vector{Pair{String,Float64}}(undef, 0))::Results where {T1<:INGARCH, T2 <: Real}
    MLEControl = MLESettings(y, model)

    fit(y, model, MLEControl, printResults=printResults, initiate=initiate, restr = restr)
end

function fit(y::Vector{Int64},
    model::T1,
    MLEControl::MLEControl;
    printResults::Bool=true,
    restr::Vector{Pair{String, T2}} = Vector{Pair{String, Float64}}(undef, 0))::Results where {T1<:INARMA, T2 <: Real}

    T = length(y)
    nR = length(restr)

    if typeof(model) in [INARMAModel, INMAModel]
        q = length(model.pastMean)
        Q = maximum(model.pastMean)

        if Q > 2
            error("INARMA(p, q) / INMA(q) with q > 2 not yet supported.")
        end
    else
        q = 0
        Q = 0
    end

    if typeof(model) != INMAModel
        p = length(model.pastObs)
        P = maximum(model.pastObs)
    else
        p = 0
        P = 0
    end

    r = length(model.external)
    M = maximum([P, Q])

    logl1 = (model.link[1] == "Log")
    logl2 = (model.link[2] == "Log")

    nb1 = model.distr[1] == "NegativeBinomial"
    nb2 = model.distr[2] == "NegativeBinomial"
    if r == 0
        nb2 = false
    else
        if sum(model.external) == 0
            nb2 = false
        end
    end

    zi = model.zi

    nObs = T - M
    nPar = 1 + p + q + r + nb1 + nb2 + zi - nR

    optimFun = BFGS()
    if MLEControl.optimizer == "LBFGS"
        optimFun = LBFGS()
    end

    if MLEControl.optimizer == "NelderMead"
        optimFun = NelderMead()
    end

    initVec = par2θ(MLEControl.init, model, restr)
    initVec = removeRestrictions(initVec, model, restr)
    
    if ll(y, model, initVec, restr)[1] == -Inf
        error("Infinite likelihood for initial parameters. Provide other values.")
    end

    optRaw = optimize(vars -> -ll(y, model, vars, restr)[1],
        initVec,
        optimFun,
        Optim.Options(iterations=MLEControl.maxEval))

    cvg = Optim.converged(optRaw)
    estsVec = optRaw.minimizer
    ests = θ2par(estsVec, model, restr)

    temp = ll(y, model, ests)
    LLmax = temp[1]
    LLs = temp[2]

    se = zeros(nPar)
    CI = zeros(2, nPar)

    ciProb = false
    if nb1 | nb2
        if any(ests.ϕ .> 10000)
            ciProb = true
            println("No confidence intervals computed due to large estimate of ϕ.")
        end
    end

    if MLEControl.ci & !ciProb
        H = Calculus.hessian(vars -> ll(y, model, vars, restr, initiate=initiate)[1], estsVec)
        if any(isnan.(H)) | any(.!isfinite.(H))
            ciProb = true
            println("No confidence intervals computed due to Inf/NaN in Hessian.")
        end
    end

    if MLEControl.ci & !ciProb
        Σ = -inv(H)
        se = sqrt.(diag(Σ))
        quan = quantile(Normal(), 0.975)
        for i = 1:nPar
            CI[:, i] = estsVec[i] .+ [-1, 1] .* se[i] .* quan
        end
    else
        Σ = zeros(nPar, nPar)
    end

    # Compute residuals
    λ = fittedValues(y, model, ests)
    resi = y .- λ

    res = INARMAresults(y, estsVec, ests, λ, resi, LLmax, LLs, nPar, nObs, se, CI, Σ, model, cvg, MLEControl)

    if printResults
        show(res, restr)
    end

    return res
end

function fit(y::Vector{Int64},
    model::T1;
    printResults::Bool=true,
    restr::Vector{Pair{String, T2}} = Vector{Pair{String, Float64}}(undef, 0))::Results where {T1<:INARMA, T2 <: Real}

    MLEControl = MLESettings(y, model)
    fit(y, model, MLEControl, printResults=printResults, restr = restr)
end

##### Old code below this line #####

# function fit(y::Vector{Int64},
#     model::T1,
#     MLEControl::MLEControl;
#     printResults::Bool=true,
#     initiate::String="first")::Results where {T1<:INGARCH}
#     T = length(y)

#     if typeof(model) == INGARCHModel
#         q = length(model.pastMean)
#         Q = maximum(model.pastMean)
#     else
#         q = 0
#         Q = 0
#     end

#     if typeof(model) != IIDModel
#         p = length(model.pastObs)
#         P = maximum(model.pastObs)
#     else
#         p = 0
#         P = 0
#     end

#     r = length(model.external)
#     M = maximum([P, Q])

#     logl = (model.link == "Log")
#     lin = !logl

#     nb = model.distr == "NegativeBinomial"

#     zi = model.zi

#     nObs = T - M
#     nPar = 1 + p + q + r + nb + zi

#     optimFun = BFGS()
#     if MLEControl.optimizer == "LBFGS"
#         optimFun = LBFGS()
#     end

#     if MLEControl.optimizer == "NelderMead"
#         optimFun = NelderMead()
#     end

#     if ll(y, model, MLEControl.init, initiate=initiate)[1] == -Inf
#         error("Infinite likelihood for initial parameters. Provide other values.")
#     end

#     initVec = par2θ(MLEControl.init, model)
#     optRaw = optimize(vars -> -ll(y, model, vars, initiate=initiate)[1],
#         initVec,
#         optimFun,
#         Optim.Options(iterations=MLEControl.maxEval))
#     cvg = Optim.converged(optRaw)
#     estsVec = optRaw.minimizer
#     ests = θ2par(estsVec, model)
#     λ = LinPred(y, model, ests)

#     temp = ll(y, model, ests, initiate=initiate)
#     LLmax = temp[1]
#     LLs = temp[2]

#     resi = y .- λ

#     se = zeros(nPar)
#     CI = zeros(2, nPar)

#     ciProb = false
#     if nb
#         if ests.ϕ[1] > 10000
#             ciProb = true
#             println("No confidence intervals computed due to large estimate of ϕ.")
#         end
#     end

#     if MLEControl.ci & !ciProb
#         H = Calculus.hessian(vars -> ll(y, model, vars, initiate=initiate)[1], estsVec)
#         Σ = -inv(H)
#         se = sqrt.(diag(Σ))
#         quan = quantile(Normal(), 0.975)
#         for i = 1:nPar
#             CI[:, i] = estsVec[i] .+ [-1, 1] .* se[i] .* quan
#         end
#     else
#         Σ = zeros(nPar, nPar)
#     end

#     res = INGARCHresults(y, estsVec, ests, λ, resi, LLmax, LLs, nPar, nObs, se, CI, Σ, model, cvg, MLEControl)

#     if printResults
#         show(res)
#     end

#     return res
# end

# function fit(y::Vector{Int64},
#     model::T;
#     printResults::Bool=true,
#     initiate::String="first")::Results where {T<:INGARCH}
#     MLEControl = MLESettings(y, model)

#     fit(y, model, MLEControl, printResults=printResults, initiate=initiate)
# end

# function fit(y::Vector{Int64},
#     model::T1,
#     MLEControl::MLEControl;
#     printResults::Bool=true)::Results where {T1<:INARMA}

#     T = length(y)

#     if typeof(model) in [INARMAModel, INMAModel]
#         q = length(model.pastMean)
#         Q = maximum(model.pastMean)

#         if Q > 2
#             error("INARMA(p, q) / INMA(q) with q > 2 not yet supported.")
#         end
#     else
#         q = 0
#         Q = 0
#     end

#     if typeof(model) != INMAModel
#         p = length(model.pastObs)
#         P = maximum(model.pastObs)
#     else
#         p = 0
#         P = 0
#     end

#     r = length(model.external)
#     M = maximum([P, Q])

#     logl1 = (model.link[1] == "Log")
#     logl2 = (model.link[2] == "Log")

#     nb1 = model.distr[1] == "NegativeBinomial"
#     nb2 = model.distr[2] == "NegativeBinomial"
#     if r == 0
#         nb2 = false
#     else
#         if sum(model.external) == 0
#             nb2 = false
#         end
#     end

#     zi = model.zi

#     nObs = T - M
#     nPar = 1 + p + q + r + nb1 + nb2 + zi

#     optimFun = BFGS()
#     if MLEControl.optimizer == "LBFGS"
#         optimFun = LBFGS()
#     end

#     if MLEControl.optimizer == "NelderMead"
#         optimFun = NelderMead()
#     end

#     if ll(y, model, MLEControl.init)[1] == -Inf
#         error("Infinite likelihood for initial parameters. Provide other values.")
#     end

#     initVec = par2θ(MLEControl.init, model)

#     optRaw = optimize(vars -> -ll(y, model, vars)[1],
#         initVec,
#         optimFun,
#         Optim.Options(iterations=MLEControl.maxEval))

#     cvg = Optim.converged(optRaw)
#     estsVec = optRaw.minimizer
#     ests = θ2par(estsVec, model)

#     temp = ll(y, model, ests)
#     LLmax = temp[1]
#     LLs = temp[2]

#     se = zeros(nPar)
#     CI = zeros(2, nPar)

#     ciProb = false
#     if nb1 | nb2
#         if any(ests.ϕ .> 10000)
#             ciProb = true
#             println("No confidence intervals computed due to large estimate of ϕ.")
#         end
#     end

#     if MLEControl.ci & !ciProb
#         H = Calculus.hessian(vars -> ll(y, model, vars)[1], estsVec)
#         Σ = -inv(H)
#         se = sqrt.(diag(Σ))
#         quan = quantile(Normal(), 0.975)
#         for i = 1:nPar
#             CI[:, i] = estsVec[i] .+ [-1, 1] .* se[i] .* quan
#         end
#     else
#         Σ = zeros(nPar, nPar)
#     end

#     res = INARMAresults(y, estsVec, ests, LLmax, LLs, nPar, nObs, se, CI, Σ, model, cvg, MLEControl)

#     if printResults
#         show(res)
#     end

#     return res
# end

# function fit(y::Vector{Int64},
#     model::T1;
#     printResults::Bool=true)::Results where {T1<:INARMA}

#     MLEControl = MLESettings(y, model)
#     fit(y, model, MLEControl, printResults=printResults)
# end