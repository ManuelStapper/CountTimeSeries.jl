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
# model:        Model specification
# converged:    Success of optimization routine

"""
    INGARCHResults(y, θ, pars, λ, residuals,
          LL, LLs, nPar, nObs, se, CI,
          model, converged, MLEControl)
    INARMAResults(y, θ, pars,
          LL, LLs, nPar, nObs, se, CI,
          model, converged, MLEControl)
Structure for estimation results.

* `y`: Time series
* `θ`: Estimates (vector)
* `pars`: Estimates (parameter)
* `λ`: Conditional means (INGARCH)
* `residuals`: Residuals (y - λ) (INGARCH)
* `LL`: Maximum of likelihood
* `LLs`: Likelihood contributions
* `nPar`: Number of parameters
* `nObs`: Number of observations
* `se`: Standard errors
* `CI`: Confidence intervals
* `model`: Model specification
* `converged`: Indicator, convergence of optimization routine?
* `MLEControl`: Estimation settings used
"""
mutable struct INGARCHresults
    y::Vector{T} where T<:Integer
    θ::Vector{T} where T<: AbstractFloat
    pars::parameter
    λ::Vector{T} where T<:AbstractFloat
    residuals::Vector{T} where T<:AbstractFloat
    LL::T where T<: AbstractFloat
    LLs::Array{T, 1} where T<:AbstractFloat
    nPar::T where T<:Integer
    nObs::T where T<: Integer
    se::Array{T, 1} where T<:AbstractFloat
    CI::Array{T, 2} where T<:AbstractFloat
    model::T where T<:INGARCH
    converged::Bool
    MLEControl::MLEControl
end

mutable struct INARMAresults
    y::Vector{T} where T<:Integer
    θ::Vector{T} where T<: AbstractFloat
    pars::parameter
    LL::T where T<: AbstractFloat
    LLs::Array{T, 1} where T<:AbstractFloat
    nPar::T where T<:Integer
    nObs::T where T<: Integer
    se::Array{T, 1} where T<:AbstractFloat
    CI::Array{T, 2} where T<:AbstractFloat
    model::T where T<:INARMA
    converged::Bool
    MLEControl::MLEControl
end
