# Goal: Functions to compute mean, variance, autocovariance
#       and autocorrelation of INGARCH/INARMA processes

# Restrict to cases without regressors and with linear link

###################
##### INGARCH #####
###################

import Distributions.mean, Distributions.var

"""
    mean(model, θ, ofλ = false)
Marginal mean of INGARCH(p, q) process.
Models with regressors or a logarithmic link are not covered.

* `model`: Model specification
* `θ`: Parameters (Vector or parameter type)
* `ofλ`: If true, the marginal mean of the conditional mean sequence is returned

# Examples
```julia-repl
model = Model(pastObs = 1)
mean(model, [10, 0.5])
```
"""
function mean(model::T,
              θ::parameter,
              ofλ::Bool = false)::Float64 where {T <: INGARCH}
    if length(model.X) > 0
        error("Function not defined for process with regressors.")
    end
    if model.link != "Linear"
        error("Function only defined for linear link.")
    end
    if model.zi
        if ofλ
            return θ.β0/(1 - (1 - θ.ω)*sum(θ.α) - sum(θ.β))
        else
            return (1 - θ.ω)*θ.β0/(1 - (1 - θ.ω)*sum(θ.α) - sum(θ.β))
        end
        
    else
        return θ.β0/(1 - sum(θ.α) - sum(θ.β))
    end
end

function mean(model::T1,
              θ::Vector{T2},
              ofλ::Bool = false)::Float64 where {T1 <: INGARCH, T2 <: Real}
    mean(model, θ2par(θ, model))    
end


### Marginal Variance and autocovariances
# Use γ(1, ..., max(p, q)) and γλ(1, ..., q) for initialisation

"""
    acvf(model, θ, lagMax, ofλ)
Autocovariance function of INGARCH(p, q) process.
Models with regressors or logarithmic link are not covered.

* `model`: Model specification
* `θ`: Parameters (Vector or parameter type)
* `lagMax`: Maximum lag to be computed
* `ofλ`: If true, return ACVF/ACF of conditional mean sequence

# Examples
```julia-repl
model = Model(pastObs = 1)
acvf(model, [10, 0.5])
```
"""
function acvf(model::T,
              θ::parameter,
              lagMax::Int64 = 10,
              ofλ::Bool = false)::Vector{Float64} where {T <: INGARCH}
    if length(model.X) > 0
        error("Function not defined for process with regressors.")
    end
    if model.link != "Linear"
        error("Function only defined for linear link.")
    end
    if typeof(model) == IIDModel
        p = 0
        q = 0
    elseif typeof(model) == INARCHModel
        p = maximum(model.pastObs)
        q = 0
    else
        p = maximum(model.pastObs)
        q = maximum(model.pastMean)
    end
    if p > length(θ.α)
        α = zeros(p)
        α[model.pastObs] = θ.α
    else
        α = θ.α
    end

    if q > length(θ.β)
        β = zeros(q)
        β[model.pastMean] = θ.β
    else
        β = θ.β
    end

    zi = model.zi
    if zi
        ω = θ.ω
    else
        ω = 0.0
    end
    nb = model.distr == "NegativeBinomial"
    gp = model.distr == "GPoisson"
    if nb | gp
        ϕ = θ.ϕ[1]
    else
        ϕ = Inf
    end

    pq = maximum([p, q])
    μ = mean(model, θ, false)
    λ = mean(model, θ, true)

    X = zeros(pq + q + 2, pq + q + 2)
    y = zeros(pq + q + 2)
    
    X[1, pq + 2] = (1 - ω)*(1/ϕ + 1)
    y[1] = -λ*(1 - ω)*(λ/ϕ + λ*ω + 1)
    if gp
        X[1, pq + 2] = 1 - ω
        y[1] = -(ϕ^2 * λ * (1 - ω) + λ^2 * ω * (1 - ω))
    end
    X[1, 1] -= 1
    

    for h = 1:pq
        for i = 1:p
            X[h+1, abs(h - i) + 1] += α[i]*(1 - ω)  
        end
        for i = 1:q
            if i <= minimum([h-1, q])
                X[h+1, abs(h - i) + 1] += β[i]
            else
                X[h+1, abs(h - i) + 2 + pq] += β[i]*(1 - ω)^2
            end
        end
        X[h+1, h+1] -= 1
    end

    for h = 0:q
        for i = 1:p
            if i <= minimum([h, p])
                X[pq + 2 + h, pq + 2 + abs(h - i)] = (1 - ω)*α[i]
            else
                X[pq + 2 + h, abs(h-i) + 1] = α[i]/(1 - ω)
            end
        end
        for i = 1:q
            X[pq + 2 + h, pq + 2 + abs(h - i)] = β[i]
        end
        X[pq + 2 + h, pq + 2 + h] -= 1
    end

    temp = inv(X)*y
    γ = zeros(lagMax + 1)
    γλ = zeros(lagMax + 1)

    γ[1:pq+1] = temp[1:pq+1]
    γλ[1:1+q] = temp[pq+2:end]

    for h = q+1:pq
        for i = 1:minimum([h, p])
            γλ[h+1] += (1 - ω)*α[i]*γλ[abs(h-i) + 1]
        end
        for i = h+1:p
            γλ[h+1] += α[i]/(1 - ω)*γ[abs(h-i) + 1]
        end
        for i = 1:q
            γλ[h+1] += β[i]*γλ[abs(h-i) + 1]
        end
    end

    for h = pq+1:lagMax
        for i = 1:p
            γλ[h+1] += (1 - ω)*α[i]*γλ[abs(h - i) + 1]
        end
        for i = 1:q
            γλ[h+1] += β[i]*γλ[abs(h - i) + 1]
        end

        for i = 1:p
            γ[h+1] += (1 - ω)*α[i]*γ[abs(h - i) + 1]
        end
        for i = 1:q
            γ[h+1] += β[i]*γ[abs(h - i) + 1]
        end
    end

    if ofλ
        return γλ
    else
        return γ
    end
end

"""
    acf(model, θ, lagMax, ofλ)
Autocorrelation function of INGARCH(p, q) process.
Models with regressors or logarithmic link are not covered.

* `model`: Model specification
* `θ`: Parameters (Vector or parameter type)
* `lagMax`: Maximum lag to be computed
* `ofλ`: If true, return ACVF/ACF of conditional mean sequence

# Examples
```julia-repl
model = Model(pastObs = 1)
acf(model, [10, 0.5])
```
"""
function acf(model::T,
    θ::parameter,
    lagMax::Int64 = 10,
    ofλ::Bool = false)::Vector{Float64} where {T <: INGARCH}
    
    raw = acvf(model, θ, lagMax, ofλ)
    return raw./raw[1]
end

"""
    var(model, θ, ofλ = false)
Marginal variance of INGARCH(p, q) process.
Models with regressors or a logarithmic link are not covered.

* `model`: Model specification
* `θ`: Parameters (Vector or parameter type)
* `ofλ`: If true, the marginal mean of the conditional mean sequence is returned

# Examples
```julia-repl
model = Model(pastObs = 1)
var(model, [10, 0.5])
```
"""
function var(model::T,
             θ::parameter,
             ofλ::Bool = false)::Float64 where {T <: INGARCH}
    if typeof(model) == INGARCHModel
        pq = maximum([model.pastObs; model.pastMean])
    elseif typeof(model) == INARCHModel
        pq = maximum(model.pastObs)
    else
        pq = 0
    end
    acvf(model, θ, pq, ofλ)[1]
end

function acvf(model::T1,
              θ::Vector{T2},
              lagMax::Int64 = 10,
              ofλ::Bool = false)::Vector{Float64} where {T1 <: INGARCH, T2 <: Real}

    acvf(model, θ2par(θ, model), lagMax, ofλ)
end

function acf(model::T1,
    θ::Vector{T2},
    lagMax::Int64 = 10,
    ofλ::Bool = false)::Vector{Float64} where {T1 <: INGARCH, T2 <: Real}

    acf(model, θ2par(θ, model), lagMax, ofλ)
end

function var(model::T1,
             θ::Vector{T2},
             ofλ::Bool = false)::Float64 where {T1 <: INGARCH, T2 <: Real}
    var(model, θ2par(θ, model), ofλ)
end


##################
##### INARMA #####
##################


"""
    mean(model, θ)
Marginal mean of INARMA(p, q) process.
Models with regressors are not covered.

* `model`: Model specification
* `θ`: Parameters (Vector or parameter type)

# Examples
```julia-repl
model = Model(model = "INARMA", pastObs = 1)
mean(model, [10, 0.5])
```
"""
function mean(model::T,
              θ::parameter)::Float64 where {T <: INARMA}
    
    if length(model.X) > 0
        error("Function not defined for process with regressors.")
    end
    if model.link[1] != "Linear"
        error("Function only defined for linear link.")
    end
    ER = θ.β0*ifelse(model.zi, 1 - θ.ω, 1)
    if typeof(model) == INARModel
        return ER/(1 - sum(θ.α))
    elseif typeof(model) == INMAModel
        return ER*(1 + sum(θ.β))
    else
        return ER*(1 + sum(θ.β))/(1 - sum(θ.α))
    end
end


function mean(model::T1,
              θ::Vector{T2})::Float64 where {T1 <: INARMA, T2 <: Real}
    mean(model, θ2par(θ, model))
end

### Marginal Variance and autocovariances
"""
    acvf(model, θ, lagMax)
Autocovariance function of INARMA(p, q) process.
Models with regressors are not covered.

* `model`: Model specification
* `θ`: Parameters (Vector or parameter type)
* `lagMax`: Maximum lag to be computed

# Examples
```julia-repl
model = Model(model = "INARMA", pastObs = 1)
acvf(model, [10, 0.5])
```
"""
function acvf(model::T,
    θ::parameter,
    lagMax::Int64 = 10)::Vector{Float64} where {T <: INARMA}
    
    if length(model.X) > 0
        error("Function not defined for process with regressors.")
    end
    if model.link[1] != "Linear"
        error("Function only defined for linear link.")
    end
    if typeof(model) == INARModel
        p = maximum(model.pastObs)
        q = 0
    elseif typeof(model) == INMAModel
        p = 0
        q = maximum(model.pastMean)
    else
        p = maximum(model.pastObs)
        q = maximum(model.pastMean)
    end
    if p > length(θ.α)
        α = zeros(p)
        α[model.pastObs] = θ.α
    else
        α = θ.α
    end
    
    β0 = θ.β0

    if q > length(θ.β)
        β = zeros(q)
        β[model.pastMean] = θ.β
    else
        β = θ.β
    end

    if q == 0
        q = 1
        β = [0.0]
    end
    if p == 0
        p = 1
        α = [0.0]
    end

    zi = model.zi
    if zi
        ω = θ.ω
    else
        ω = 0.0
    end
    nb = model.distr[1] == "NegativeBinomial"
    gp = model.distr[1] == "GPoisson"
    if nb | gp
        ϕ = θ.ϕ[1]
    else
        ϕ = Inf
    end

    # Compute closed-form moments
    λ = (1 - ω)*β0
    μ = λ*(1 + sum(β))/(1 - sum(α))
    VarR = λ + β0^2*(1 - ω)*(1/ϕ + ω)
    if gp
        VarR = ϕ^2*β0*(1 - ω) + β0^2*ω*(1 - ω)
    end

    X = zeros(p + q, p + q)
    y = zeros(p + q)

    # Step 1: Include formula for γ(0)
    for i = 1:p, j = 1:p
        ind = abs(i - j) + 1
        X[1, ind] += α[i] * α[j]
    end
    X[1, p+1] += 1 + sum(β.^2)

    for i = 1:q, j = 1:p
        if i >= j
            ind = p + abs(i - j) + 1
            X[1, ind] += 2*α[j]*β[i]
        end 
    end

    y[1] = λ*sum(β .* (1 .- β)) + μ*sum(α .* (1 .- α))

    # Step 2: Include γ(h) for h > 1
    for h = 1:p-1
        for i = 1:p
            X[h+1, abs(h-i) + 1] += α[i]
        end
        for i = h:q
            X[h+1, p + 1 + i - h] += β[i]
        end
    end

    # Step 3: Include γR(0)
    y[p + 1] = VarR

    # Step 4: Include γR(h) for h > 1
    for h = 1:q-1
        for i = 1:minimum([h, p])
            X[p + h + 1, p + 1 + h - i] += α[i] 
        end
        if h <= q
            X[p + h + 1, p + 1] += β[h]
        end
    end

    res = inv(diagm(ones(p + q)) .- X)*y
    γ = res[1:p]
    γR = res[q+1:end]

    # Initialisation done, now going forward
    out = zeros(lagMax + 1)
    out[1:length(γ)] = γ

    for h = p:lagMax
        for i = h:q
            if (i - h) >= 0
                out[h + 1] += β[i]*γR[i - h + 1] 
            end
        end
        for i = 1:p
            out[h + 1] += α[i]*out[h - i + 1]
        end
    end

    return out
end

"""
    acf(model, θ, lagMax)
Autocorrelation function of INARMA(p, q) process.
Models with regressors are not covered.

* `model`: Model specification
* `θ`: Parameters (Vector or parameter type)
* `lagMax`: Maximum lag to be computed

# Examples
```julia-repl
model = Model(model = "INARMA", pastObs = 1)
acf(model, [10, 0.5])
```
"""
function acf(model::T,
             θ::parameter,
             lagMax::Int64 = 10)::Vector{Float64} where {T <: INARMA}
    
    raw = acvf(model, θ, lagMax)
    return raw./raw[1]
end

"""
    var(model, θ)
Marginal variance of INARMA(p, q) process.
Models with regressors are not covered.

* `model`: Model specification
* `θ`: Parameters (Vector or parameter type)

# Examples
```julia-repl
model = Model(model = "INARMA", pastObs = 1)
var(model, [10, 0.5])
```
"""
function var(model::T,
             θ::parameter)::Float64 where {T <: INARMA}
    if typeof(model) == INARMAModel
        pq = maximum([model.pastObs; model.pastMean])
    elseif typeof(model) == INARModel
        pq = maximum(model.pastObs)
    else
        pq = maximum(model.pastMean)
    end
    acvf(model, θ, pq)[1]
end

function acvf(model::T1,
              θ::Vector{T2},
              lagMax::Int64 = 10)::Vector{Float64} where {T1 <: INARMA, T2 <: Real}
    acvf(model, θ2par(θ, model), lagMax)
end

function acf(model::T1,
    θ::Vector{T2},
    lagMax::Int64 = 10)::Vector{Float64} where {T1 <: INARMA, T2 <: Real}

    acf(model, θ2par(θ, model), lagMax)
end

function var(model::T1,
             θ::Vector{T2})::Float64 where {T1 <: INARMA, T2 <: Real}
    var(model, θ2par(θ, model))
end