"""
    CountModel
Abstract type which summarizes all models in the CountTimeSeries.jl package.
"""
abstract type CountModel end

"""
    INGARCH
Abstract type which summarizes all models in the INGARCH framework. Subtype of CountModel.
"""
abstract type INGARCH<:CountModel end

"""
    INARMA
Abstract type which summarizes all models in the INARMA framework. Subtype of CountModel.
"""
abstract type INARMA<:CountModel end


"""
    INGARCHModel
Struct to specify an INGARCH model. It contains information on
* `distr`: Distribution
* `link`: Link function
* `pastObs`: Past observations included in the conditional mean definition
* `pastMean`: Past mean included in the conditional mean definition
* `X`: Regressor matrix
* `external`: Indicator which regressors enter the system externally
* `zi`: Indicator. Is zero inflation modelled?
"""
mutable struct INGARCHModel<:INGARCH
    distr::String
    link::String
    pastObs::Array{T, 1} where T<:Integer
    pastMean::Array{T, 1} where T<:Integer
    X::Array{T, 2} where T<: AbstractFloat
    external::Array{Bool, 1}
    zi::Bool
end

"""
    INARCHModel
Struct to specify an INARCH model. It contains information on
* `distr`: Distribution
* `link`: Link function
* `pastObs`: Past observations included in the conditional mean definition
* `X`: Regressor matrix
* `external`: Indicator which regressors enter the system externally
* `zi`: Indicator. Is zero inflation modelled?
"""
mutable struct INARCHModel<:INGARCH
    distr::String
    link::String
    pastObs::Array{T, 1} where T<:Integer
    X::Array{T, 2} where T<: AbstractFloat
    external::Array{Bool, 1}
    zi::Bool
end

"""
    IIDModel
Struct to specify an IID model. It contains information on
* `distr`: Distribution
* `link`: Link function
* `X`: Regressor matrix
* `external`: Indicator which regressors enter the system externally
* `zi`: Indicator. Is zero inflation modelled?
"""
mutable struct IIDModel<:INGARCH
    distr::String
    link::String
    X::Array{T, 2} where T<: AbstractFloat
    external::Array{Bool, 1}
    zi::Bool
end

"""
    INARMAModel
Struct to specify an INARMA model. It contains information on
* `distr`: Distributions (vector)
* `link`: Link functions (vector)
* `pastObs`: Past observations included
* `pastMean`: Past mean included
* `X`: Regressor matrix
* `external`: Indicator which regressors enter the system externally
* `zi`: Indicator. Is zero inflation modelled?
"""
mutable struct INARMAModel<:INARMA
    distr::Array{String, 1}
    link::Array{String, 1}
    pastObs::Array{T, 1} where T<:Integer
    pastMean::Array{T, 1} where T<:Integer
    X::Array{T, 2} where T<: AbstractFloat
    external::Array{Bool, 1}
    zi::Bool
end

"""
    INARModel
Struct to specify an INAR model. It contains information on
* `distr`: Distributions (vector)
* `link`: Link functions (vector)
* `pastObs`: Past observations included
* `X`: Regressor matrix
* `external`: Indicator which regressors enter the system externally
* `zi`: Indicator. Is zero inflation modelled?
"""
mutable struct INARModel<:INARMA
    distr::Array{String, 1}
    link::Array{String, 1}
    pastObs::Array{T, 1} where T<:Integer
    X::Array{T, 2} where T<: AbstractFloat
    external::Array{Bool, 1}
    zi::Bool
end

"""
    INMAModel
Struct to specify an INMA model. It contains information on
* `distr`: Distributions (vector)
* `link`: Link functions (vector)
* `pastMean`: Past mean included
* `X`: Regressor matrix
* `external`: Indicator which regressors enter the system externally
* `zi`: Indicator. Is zero inflation modelled?
"""
mutable struct INMAModel<:INARMA
    distr::Array{String, 1}
    link::Array{String, 1}
    pastMean::Array{T, 1} where T<:Integer
    X::Array{T, 2} where T<: AbstractFloat
    external::Array{Bool, 1}
    zi::Bool
end
