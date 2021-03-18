# Struct for model parameters

"""
    parameter(β0, α, β, η, ϕ, ω)
Structure for Count Data models parameters.

* `β0`: Intercept parameter
* `α`: Parameters for autoregression
* `β`: Parameters for MA part/past means
* `η`: Regressor parameters
* `ϕ`: Overdispersion parameter (Negative Binomial)
* `ω`: Zero inflation probability

For details, see For details, see [Documentation](https://github.com/ManuelStapper/CountTimeSeries.jl/blob/master/CountTimeSeries_documentation.pdf)
"""
mutable struct parameter
    β0::T where T<:AbstractFloat
    α::Array{T, 1} where T<:AbstractFloat
    β::Array{T, 1} where T<:AbstractFloat
    η::Array{T, 1} where T<:AbstractFloat
    ϕ::Array{T, 1} where T<:AbstractFloat
    ω::T where T<:AbstractFloat
end
