# Implementation of the Generalised Poisson Distribution

"""
    GPoisson(λ::Real, ϕ::Real)
    
The Generalised Poisson distribution with parameters `λ` and `ϕ`, see [Article](https://doi.org/10.1016/j.jmaa.2011.11.042)

# Example
```julia
d = GPoisson(5, 0.7)
mean(d)
pdf(d, 1)
```
"""
struct GPoisson{T<:Real} <: DiscreteUnivariateDistribution
    λ::T
    ϕ::T
    GPoisson{T}(λ::T, ϕ::T) where {T} = new{T}(λ, ϕ)
end

import Distributions.@check_args
function GPoisson(λ::T, ϕ::T; check_args=true) where {T<:Real}
    check_args && @check_args(GPoisson, (λ >= zero(λ) && ϕ > zero(ϕ)))
    return GPoisson{T}(λ, ϕ)
end

GPoisson(λ::Integer, ϕ::Integer) = GPoisson(float(λ), float(ϕ))
GPoisson(λ::Integer, ϕ::Float64) = GPoisson(float(λ), ϕ)
GPoisson(λ::Float64, ϕ::Integer) = GPoisson(λ, float(ϕ))


import Distributions.@distr_support
import Base.minimum, Base.maximum, Distributions.minimum, Distributions.maximum
@distr_support GPoisson 0 (d.λ == zero(typeof(d.λ)) ? 0 : ifelse(d.ϕ < 1, floor(d.λ/(1 - d.ϕ)), Inf))
# function minimum(d::GPoisson)
#    return 0
# end
# function maximum(d::GPoisson)
#     return ifelse(d.ϕ < 1, floor(d.λ/(1 - d.ϕ)), Inf)
# end

#### Conversions

import Base.convert
convert(::Type{GPoisson{T}}, λ::S, ϕ::S) where {T<:Real,S<:Real} = GPoisson(T(λ), T(ϕ))
convert(::Type{GPoisson{T}}, d::GPoisson{S}) where {T<:Real,S<:Real} = GPoisson(T(d.λ), T(d.ϕ), check_args=false)

### Parameters
import Distributions.params
params(d::GPoisson) = (d.λ, d.ϕ)
import Distributions.partype
partype(::GPoisson{T}) where {T} = T

import Distributions.mean
function mean(d::GPoisson)
    (λ, ϕ) = params(d)
    return λ
end

import Distributions.mode
function mode(d::GPoisson)
    out = 0
    pmfMax = pdf(d, 0)
    while true
        pmfNew = pdf(d, out + 1)
        if pmfNew > pmfMax
            out += 1
            pmfMax = pmfNew
        else
            return out
        end
    end
end

import Distributions.modes
function modes(d::GPoisson)
    x = 0
    out = [0]
    pmfMax = pdf(d, 0)
    while true
        cand = x + 1
        pmfNew = pdf(d, cand)
        if pmfNew > pmfMax
            out = [cand]
            pmfMax = pmfNew
            x += 1
        elseif pmfNew == pmfMax
            out = [out; cand]
            x += 1
        else
            return out
        end
    end
end

import Distributions.var
function var(d::GPoisson)
    (λ, ϕ) = params(d)
    return ϕ^2*λ
end

import Distributions.pdf, Distributions.logpdf
function pdf(d::GPoisson, x::Integer)
    (λ, ϕ) = params(d)
    if λ == 0
        return ifelse(x == 0, 1.0, 0.0)
    end
    κ = (ϕ - 1)/ϕ
    λ2 = λ*(1 - κ)
    if κ < 0
        m = floor(-λ2/κ)
    else
        m = Inf
    end

    if x > m
        return 0.0
    end

    if (λ2 + κ*x <= 0) | (λ2 <= 0)
        return 0.0
    end
    exp(log(λ2) + (x-1)*log(λ2 + κ*x) - (λ2 + κ*x) - logabsgamma(x+1)[1])
end

function logpdf(d::GPoisson, x::T) where {T <: Real}
    return log(pdf(d, x))
end

function pdf(d::GPoisson, x::Real)
    round(Int, x) == x ? pdf(d, round(Int, x)) : 0.0
end

import Distributions.cdf
function cdf(d::GPoisson, x::Integer)
    if x < 0
        return 0.0
    end
    return sum((xx -> pdf(d, xx)).(0:x))
end

function cdf(d::GPoisson, x::Real)
    if x < 0
        return 0.0
    end
    cdf(d, floor(Int, x))
end

# Random Number Generation
import Distributions.rand
function rand(d::GPoisson)
    out = 0
    pmfSum = pdf(d, 0)
    u = rand()
    maxVal = maximum(d)
    while (pmfSum < u) & (out < maxVal)
        out += 1
        pmfSum += pdf(d, out)
    end
    out
end

function rand(d::GPoisson, n::Int64)
    out = zeros(Int64, n) .- 1
    counter = 0
    u = rand(n)
    tbd = fill(true, n)
    uMax = maximum(u)
    pmfSum = pdf(d, 0)
    maxVal = maximum(d)
    while (pmfSum < uMax) & (counter < maxVal)
        out[(u .<= pmfSum) .& tbd] .= counter
        tbd[u .<= pmfSum] .= false
        counter += 1
        pmfSum += pdf(d, counter)
    end
    out[tbd] .= counter
    out
end

function rand(d::GPoisson, dims::Dims)
    x = rand(d, prod(dims))
    reshape(x, dims)
end

function rand(d::GPoisson, dim1::Int, moredims::Int...)
    x = rand(d, dim1 * prod(moredims))
    reshape(x, (dim1, moredims...))
end


import Distributions.quantile
function quantile(d::GPoisson, u::T) where {T <: Real}
    if (u < 0) | (u > 1)
        error("Invalid probability")
    end
    if u == 1
        return maximum(d)
    end
    out = 0
    pmfSum = pdf(d, 0)
    maxVal = maximum(d)
    while (pmfSum < u) & (out < maxVal)
        out += 1
        pmfSum += pdf(d, out)
    end
    out
end