"""
    Model(model, distr, link, pastObs,
          pastMean, X, external, zi)
Structure to specify count data models with subtypes for more specific models.

* `model`: "INARMA" or "INGARCH"
* `distr`: "Poisson" or "Negative Binomial" (Vector for INARMA)
* `link`: Vector of length two, "Linear" or "Log" (Vector for INARMA)
* `pastObs`: Lags considered in autoregressive part
* `pastMean`: Lags considered in MA/past conditional mean part
* `X`: Matrix of regressors (row-wise)
* `external`: Indicator(s) if regressors enter externally
* `zi`: Indicator, zero inflation Y/N

For details, see [Documentation](https://github.com/ManuelStapper/CountTimeSeries.jl/blob/master/CountTimeSeries_documentation.pdf)
"""
abstract type CountModel end
abstract type INGARCH<:CountModel end
abstract type INARMA<:CountModel end

mutable struct INGARCHModel<:INGARCH
    distr::String
    link::String
    pastObs::Array{T, 1} where T<:Integer
    pastMean::Array{T, 1} where T<:Integer
    X::Array{T, 2} where T<: AbstractFloat
    external::Array{Bool, 1}
    zi::Bool
end

mutable struct INARCHModel<:INGARCH
    distr::String
    link::String
    pastObs::Array{T, 1} where T<:Integer
    X::Array{T, 2} where T<: AbstractFloat
    external::Array{Bool, 1}
    zi::Bool
end

mutable struct IIDModel<:INGARCH
    distr::String
    link::String
    X::Array{T, 2} where T<: AbstractFloat
    external::Array{Bool, 1}
    zi::Bool
end

mutable struct INARMAModel<:INARMA
    distr::Array{String, 1}
    link::Array{String, 1}
    pastObs::Array{T, 1} where T<:Integer
    pastMean::Array{T, 1} where T<:Integer
    X::Array{T, 2} where T<: AbstractFloat
    external::Array{Bool, 1}
    zi::Bool
end

mutable struct INARModel<:INARMA
    distr::Array{String, 1}
    link::Array{String, 1}
    pastObs::Array{T, 1} where T<:Integer
    X::Array{T, 2} where T<: AbstractFloat
    external::Array{Bool, 1}
    zi::Bool
end

mutable struct INMAModel<:INARMA
    distr::Array{String, 1}
    link::Array{String, 1}
    pastMean::Array{T, 1} where T<:Integer
    X::Array{T, 2} where T<: AbstractFloat
    external::Array{Bool, 1}
    zi::Bool
end
