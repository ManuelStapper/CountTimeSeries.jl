# Function to obtain the conditional means for given time series,
# parameters, and model specification

# Input:
# y:        Time series
# model:    Model specification
# θ:        Parameters
# X:        Matrix of regressors


# Output:
# λ:        Conditional mean

# Core operations are put in functions for speedup

function CoreIID(β0::Float64,
                 logl::Bool,
                 T::Int64)::Vector{Float64}
    if logl
        return fill(exp(β0), T)
    else
        return fill(Float64(β0), T)
    end
end

function CoreIID(β0::Float64,
                 X::Matrix{Float64},
                 η::Vector{Float64},
                 logl::Bool,
                 T::Int64)::Vector{Float64}
    out = fill(β0, T)
    out += (η'X)[1, :]
    if logl
        return exp.(out)
    else
        return out
    end
end

function LinPred(y::Array{Int64, 1},
                 model::IIDModel,
                 θ::parameter,
                 initiate::String = "first")::Vector{Float64}
    if !(initiate in ["first", "intercept", "marginal"])
        println("Initiation not specified correctly - Changed to 'fist'")
        initiate = "first"
    end

    T = length(y)
    r = length(model.external)

    logl = (model.link == "Log")
    β0 = Float64(θ.β0)

    if r > 0
        X = Matrix{Float64}(model.X)
        η = Vector{Float64}(θ.η)
        return CoreIID(β0, X, η, logl, T)
    else
        return CoreIID(β0, logl, T)
    end
end

function CoreARCH(y::Vector{Int64},
                  β0::Float64,
                  α::Vector{Float64},
                  pastObs::Vector{Int64},
                  P::Int64,
                  logl::Bool,
                  T::Int64)::Vector{Float64}
    out = fill(β0, T)
    if logl
        y = @. log(y + 1)
    end
    @inbounds for i = 1:length(pastObs)
        out[P+1:end] .+= α[i] .* y[P+1-pastObs[i]:end-pastObs[i]]
    end
    if logl
        return exp.(out)
    else
        return out
    end
end

function CoreARCH(y::Vector{Int64},
                  β0::Float64,
                  α::Vector{Float64},
                  pastObs::Vector{Int64},
                  P::Int64,
                  η::Vector{Float64},
                  X::Matrix{Float64},
                  logl::Bool,
                  T::Int64)::Vector{Float64}
    out = fill(β0, T)
    if logl
        y = @. log(y + 1)
    end
    @inbounds for i = length(pastObs)
        out[P+1:end] .+= α[i] .* y[P+1-pastObs[i]:end-pastObs[i]]
    end
    out += (η'*X)[1, :]

    if logl
        return exp.(out)
    else
        return out
    end
end

function LinPred(y::Array{Int64, 1},
                 model::INARCHModel,
                 θ::parameter,
                 initiate::String = "first")::Vector{Float64}
    if !(initiate in ["first", "intercept", "marginal"])
        println("Initiation not specified correctly - Changed to 'fist'")
        initiate = "first"
    end

    T = length(y)

    p = length(model.pastObs)
    if p == 0
        P = 0
    else
        P = maximum(model.pastObs)
    end
    r = length(model.external)

    logl = (model.link == "Log")
    lin = !logl

    β0 = Float64(θ.β0)
    α = Vector{Float64}(θ.α)

    if r > 0
        X = Matrix{Float64}(model.X)
        η = Vector{Float64}(θ.η)
        return CoreARCH(y, β0, α, model.pastObs, P, η, X, logl, T)
    else
        return CoreARCH(y, β0, α, model.pastObs, P, logl, T)
    end
end

# No regressors
function CoreGARCH(y::Vector{Int64},
                   λinit::Float64,
                   β0::Float64,
                   α::Vector{Float64},
                   pastObs::Vector{Int64},
                   p::Int64,
                   β::Vector{Float64},
                   pastMean::Vector{Int64},
                   q::Int64,
                   M::Int64,
                   logl::Bool,
                   T::Int64)::Vector{Float64}
    out = fill(β0, T)
    out[1:M] .= λinit

    if logl
        y = log.(y .+ 1)
    end
    @inbounds for t = M+1:T
        for i = 1:p
            out[t] += α[i] * y[t-pastObs[i]]
        end

        for i = 1:q
            out[t] += β[i] * out[t - pastMean[i]]
        end
    end

    if logl
        return exp.(out)
    else
        return out
    end
end

# Only Internal regressors
function CoreGARCH(y::Vector{Int64},
                   λinit::Float64,
                   β0::Float64,
                   α::Vector{Float64},
                   pastObs::Vector{Int64},
                   p::Int64,
                   β::Vector{Float64},
                   pastMean::Vector{Int64},
                   q::Int64,
                   M::Int64,
                   ηI::Vector{Float64},
                   rI::Int64,
                   XI::Matrix{Float64},
                   logl::Bool,
                   T::Int64)::Vector{Float64}
    out = fill(β0, T)
    out[1:M] .= λinit

    if logl
        y = log.(y .+ 1)
    end
    @inbounds for t = M+1:T
        for i = 1:p
            out[t] += α[i] * y[t-pastObs[i]]
        end

        for i = 1:q
            out[t] += β[i] * out[t - pastMean[i]]
        end

        for i = 1:rI
            out[t] += ηI[i] * XI[i, t]
        end
    end

    if logl
        return exp.(out)
    else
        return out
    end
end

# Both, internal and external regressors
function CoreGARCH(y::Vector{Int64},
                   λinit::Float64,
                   β0::Float64,
                   α::Vector{Float64},
                   pastObs::Vector{Int64},
                   p::Int64,
                   β::Vector{Float64},
                   pastMean::Vector{Int64},
                   q::Int64,
                   M::Int64,
                   ηI::Vector{Float64},
                   rI::Int64,
                   XI::Matrix{Float64},
                   logl::Bool,
                   T::Int64,
                   ηE::Vector{Float64},
                   XE::Matrix{Float64})::Vector{Float64}
    out = fill(β0, T)
    out[1:M] .= λinit

    if logl
        y = log.(y .+ 1)
    end
    @inbounds for t = M+1:T
        for i = 1:p
            out[t] += α[i] * y[t-pastObs[i]]
        end

        for i = 1:q
            out[t] += β[i] * out[t - pastMean[i]]
        end

        for i = 1:rI
            out[t] += ηI[i] * XI[i, t]
        end
    end

    if logl
        out = exp.(out)
    end

    return out + (ηE'XE)[1, :]
end

# Only external regressors
function CoreGARCH(y::Vector{Int64},
                   λinit::Float64,
                   β0::Float64,
                   α::Vector{Float64},
                   pastObs::Vector{Int64},
                   p::Int64,
                   β::Vector{Float64},
                   pastMean::Vector{Int64},
                   q::Int64,
                   M::Int64,
                   logl::Bool,
                   T::Int64,
                   ηE::Vector{Float64},
                   XE::Matrix{Float64})::Vector{Float64}
    out = fill(β0, T)
    out[1:M] .= λinit

    if logl
        y = log.(y .+ 1)
    end
    @inbounds for t = M+1:T
        for i = 1:p
            out[t] += α[i] * y[t-pastObs[i]]
        end

        for i = 1:q
            out[t] += β[i] * out[t - pastMean[i]]
        end
    end

    if logl
        out = exp.(out)
    end

    return out + (ηE'XE)[1, :]
end

function LinPred(y::Array{Int64,1},
                 model::INGARCHModel,
                 θ::parameter,
                 initiate::String = "first")::Vector{Float64}
    if !(initiate in ["first", "intercept", "marginal"])
        println("Initiation not specified correctly - Changed to 'fist'")
        initiate = "first"
    end

    T = length(y)
    p = length(model.pastObs)
    if p == 0
        P = 0
    else
        P = Int64(maximum(model.pastObs))
    end
    q = length(model.pastMean)
    if q == 0
        Q = 0
    else
        Q = Int64(maximum(model.pastMean))
    end

    M = Int64(maximum([P, Q]))
    r = length(model.external)

    logl = (model.link == "Log")
    lin = !logl

    β0 = Float64(θ.β0)
    α = Vector{Float64}(θ.α)
    β = Vector{Float64}(θ.β)

    if initiate == "first"
        λinit = Float64(y[1])
        if logl
            λinit = log(λinit + 1)
        end
    end
    if initiate == "marginal"
        λinit = β0/(1 - sum(α) - sum(β))
        if logl
            λinit = log(λinit)
        end
    end
    if initiate == "intercept"
        λinit = β0
        if logl
            λinit = log(λinit)
        end
    end

    if r > 0
        iE = findall(model.external)
        iI = setdiff(1:r, iE)
        rE = length(iE)
        rI = length(iI)
        X = Matrix{Float64}(model.X)
        η = Vector{Float64}(θ.η)
        if rE == r
            return CoreGARCH(y, λinit, β0, α, model.pastObs, p, β, model.pastMean, q, M, logl, T, η, X)
        end
        if rI == r
            return CoreGARCH(y, λinit, β0, α, model.pastObs, p, β, model.pastMean, q, M, η, r, X, logl, T)
        end

        return CoreGARCH(y, λinit, β0, α, model.pastObs, p, β, model.pastMean, q, M, η[iI], rI, X[iI, :], logl, T, η[iE], X[iE, :])
    else
        return CoreGARCH(y, λinit, β0, α, model.pastObs, p, β, model.pastMean, q, M, logl, T)
    end
end
