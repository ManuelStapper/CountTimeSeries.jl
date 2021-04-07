using Distributions, Optim, LinearAlgebra, Random, Plots

# Define Thinning Operators
import Base.∘

# Standard Binomial Thinning
"""
    ∘
Thinning operator for p∘X, d∘X or p∘d.

* `p`: Thinning probability (∈ [0, 1])
* `X`: Positive integer
* `d`: Discrete distribution

Random numbers are generated according to (binomial) thinning.

`p∘X` computes ``\\sum_{i = 1}^X{Z_i}`` with ``Z_i\\sim\\text{Bin}(1, p)``

`d∘X` computes ``\\sum_{i = 1}^X{Z_i}`` with ``Z_i\\sim`` `d`

`p∘d` computes ``\\sum_{i = 1}^X{Z_i}`` with ``Z_i\\sim\\text{Bin}(1, n)`` and ``X\\sim`` `d`
"""
p::Float64 ∘ X::Integer = begin
    if (0 <= p <= 1) & (X >= 0)
        rand(Binomial(X, p))
    else
        if !(0 <= p <= 1)
            error("Invalid thinning probability.")
        end
        if X < 0
            error("X must be a positive integer.")
        end
    end
end

# Generic Thinning for any distribution as summands
∘(d::DiscreteDistribution, X::Integer) = sum(rand(d, X))

# Binomial Thinning with distribution for sum limit
p::Float64 ∘ d::DiscreteDistribution = begin
    if (0 <= p <= 1)
        rand(Binomial(rand(d), p))
    else
        error("Invalid thinning probability.")
    end
end
# Generic thinning with distribution for sum limit
dInner::Distribution ∘ dOuter::DiscreteDistribution = sum(rand(dInner, rand(dOuter)))

# CRN thinning (Binomial)
function ∘(p::Float64, X::Integer, u::Float64)
    if !(0 <= p <= 1)
        error("Invalid thinning probability.")
    end

    if X < 0
        error("X must be a positive integer.")
    end

    if !(0 <= u <= 1)
        error("Invalid CRN.")
    end

    quantile(Binomial(X, p), u)
end

# CRN thinning (generic)
function ∘(d::Distribution, X::Integer, u::Vector{Float64})
    if !all(0 .<= u .<= 1)
        error("Invalid CRN.")
    end

    if X < 0
        error("X must be a positive integer.")
    end

    if length(u) < X
        error("Too few CRNs.")
    end

    sum(quantile.(d, u[1:X]))
end
