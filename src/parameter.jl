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
    β0::Float64
    α::Vector{Float64}
    β::Vector{Float64}
    η::Vector{Float64}
    ϕ::Vector{Float64}
    ω::Float64
end
