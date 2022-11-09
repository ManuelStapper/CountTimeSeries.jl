# Structured results of esimation

# y:            Time series
# θ:            Estimates
# pars:         Estimates in parameter struct
# λ:            Fitted values
# residuals:    Residuals (for INARMA using most probable sequence of R)
# LL:           Maximum value of log-likelihood
# LLs:          Single log-likelihood contributions
# nPar:         Number of parameters
# nObs:         Number of Observations
# se:           Standard errors
# CI:           Confidence intervals (normal approximation)
# Σ:            Covariance matrix of estimator
# model:        Model specification
# converged:    Success of optimization routine

abstract type Results end

"""
    INGARCHResults(y, θ, pars, λ, residuals,
          LL, LLs, nPar, nObs, se, CI, Σ,
          model, converged, MLEControl)
Structure for estimation results.

* `y`: Time series
* `θ`: Estimates (vector)
* `pars`: Estimates (parameter)
* `λ`: Conditional means
* `residuals`: Residuals (y - λ)
* `LL`: Maximum of likelihood
* `LLs`: Likelihood contributions
* `nPar`: Number of parameters
* `nObs`: Number of observations
* `se`: Standard errors
* `CI`: Confidence intervals
* `Σ`: Covariance matrix of estimator
* `model`: Model specification
* `converged`: Indicator, convergence of optimization routine?
* `MLEControl`: Estimation settings used
"""
mutable struct INGARCHresults{T <: INGARCH} <: Results
    y::Vector{Int64}
    θ::Vector{Float64}
    pars::parameter
    λ::Vector{Float64}
    residuals::Vector{Float64}
    LL::Float64
    LLs::Vector{Float64}
    nPar::Int64
    nObs::Int64
    se::Vector{Float64}
    CI::Array{Float64, 2}
    Σ::Matrix{Float64}
    model::T
    converged::Bool
    MLEControl::MLEControl
end

"""
    INARMAResults(y, θ, pars,
      LL, LLs, nPar, nObs, se, CI, Σ,
      model, converged, MLEControl)

* `y`: Time series
* `θ`: Estimates (vector)
* `pars`: Estimates (parameter)
* `LL`: Maximum of likelihood
* `LLs`: Likelihood contributions
* `nPar`: Number of parameters
* `nObs`: Number of observations
* `se`: Standard errors
* `CI`: Confidence intervals
* `Σ`: Covariance matrix of estimator
* `model`: Model specification
* `converged`: Indicator, convergence of optimization routine?
* `MLEControl`: Estimation settings used
"""
mutable struct INARMAresults{T <: INARMA} <: Results
    y::Vector{Int64}
    θ::Vector{Float64}
    pars::parameter
    LL::Float64
    LLs::Vector{Float64}
    nPar::Int64
    nObs::Int64
    se::Vector{Float64}
    CI::Array{Float64, 2}
    Σ::Matrix{Float64}
    model::T
    converged::Bool
    MLEControl::MLEControl
end
