# Wrapper function(s) to simulate from a Count Data Model
# Only check if number of parameters matches length of θ

"""
    simulate(T, model, θ; burnin, pinfirst)
Function to generate time series from Count Data model.

* `T`: Length of time series
* `model`: Model specification
* `θ`: Parameters (vector or struct)
* `burnin`: Number of burnin observations, default 500
* `pinfirst`: Set values for first observations (instead of burnin)

# Example
```julia-repl
model = CountModel(pastObs = 1)
simulate(100, model, [10, 0.5])
```
"""
function simulate(T::Int64,
                  model::T1,
                  θ::parameter;
                  burnin::Int64 = 500,
                  pinfirst::Array{T2, 1} = Vector{Float64}(undef, 0)) where {T1 <: INGARCH, T2 <: Real}
    if !parametercheck(θ, model)
        error("Parameters not valid.")
    end

    y = fill(0, T + burnin)
    X = model.X

    p = length(model.pastObs)
    q = length(model.pastMean)
    if p == 0
        P = 0
    else
        P = maximum(model.pastObs)
    end
    if q == 0
        Q = 0
    else
        Q = maximum(model.pastMean)
    end

    r = length(model.external)
    M = maximum([P, Q])

    if (size(X)[2] > 0) & (size(X)[2] != T)
        error("Number of regressors does not match T.")
    end

    logl = (model.link == "Log")
    lin = !logl

    nb = model.distr == "NegativeBinomial"
    gp = model.distr == "GPoisson"
    zi = model.zi

    β0 = θ.β0
    α = θ.α
    β = θ.β
    η = θ.η
    ϕ = θ.ϕ
    ϕ = θ.ϕ
    ω = θ.ω

    rI = 0
    rE = 0
    if r > 0
        iE = findall(model.external)
        iI = setdiff(1:r, rE)
        rE = sum(model.external)
        rI = r - rE
    end

    ν = fill(β0, T + burnin)
    λ = fill(ifelse(lin, β0, exp(β0)), T + burnin)

    if p == 0
        if (q != 0) & (r == 0)
            error("INGARCH(0, q) without regressors not allowed.")
        end

        for i = 1:r
            ν[burnin + 1:end] .+= η[i]*X[i, :]
        end
        if logl
            λ = exp.(ν)
        end

        if nb
            y = rand.(NegativeBinomial.(ϕ[1], ϕ[1]./(ϕ[1] .+ λ)))
        elseif gp
            y = rand.(GPoisson(λ, ϕ[1]))
        else
            y = rand.(Poisson.(λ))
        end


        y = y.*(rand(T + burnin) .>= ω)
        λzi = λ.*(1 - ω)

        return y[burnin + 1:end], ν[burnin + 1:end], λ[burnin + 1:end], λzi[burnin + 1:end]
    end

    if p + q > 0
        νinit = β0/(1 - sum([α; β]))
    else
        vinit = β0
    end
    λinit = ifelse(logl, exp(νinit), νinit)

    ν[1:M] = fill(νinit, M)
    λ[1:M] = fill(λinit, M)

    if nb
        y[1:M] = rand.(NegativeBinomial.(ϕ[1], ϕ[1]./(ϕ[1] .+ λ[1:M])))
    elseif gp
        y[1:M] = rand.(GPoisson.(λ[1:M], ϕ[1]))
    else
        y[1:M] = rand.(Poisson.(λ[1:M]))
    end

    if zi
        y[1:M] = y[1:M].*(rand(M) .>= ω)
    end

    for t = (M+1):(T + burnin)
        ν[t] = β0
        if logl
            ν[t] += dot(α, log.(y[t .- model.pastObs] .+ 1))
        else
            ν[t] += dot(α, y[t .- model.pastObs])
        end

        if q > 0
            ν[t] += dot(β, ν[t .- model.pastMean])
        end

        if rI > 0
            ν[t] += dot(η[iI], X[iI, maximum([1, t - burnin])])
        end

        λ[t] = ν[t]

        if rE > 0
            λ[t] += dot(η[iE], X[iE, maximum([1, t - burnin])])
        end

        if logl
            λ[t] = exp(λ[t])
        end

        if nb
            y[t] = rand(NegativeBinomial(ϕ[1], ϕ[1]/(ϕ[1] + λ[t])))
        elseif gp
            y[t] = rand(GPoisson(λ[t], ϕ[1]))
        else
            y[t] = rand(Poisson(λ[t]))
        end

        if zi
            y[t] = y[t]*(rand() >= ω)
        end

        if length(pinfirst) > 0
            if burnin + 1 <= t <= burnin + length(pinfirst)
                y[t] = pinfirst[t - burnin]
                λ[t] = pinfirst[t - burnin]
                ν[t] = ifelse(logl, log(pinfirst[t - burnin] + 1), pinfirst[t - burnin])
            end
        end
    end

    λzi = λ.*(1 - ω)

    return y[burnin + 1:end], ν[burnin + 1:end], λ[burnin + 1:end], λzi[burnin+1:end]
end

function simulate(T::Int64,
                  model::INARCHModel,
                  θ::parameter;
                  burnin::Int64 = 500 ,
                  pinfirst::Array{T1, 1} = Vector{Float64}(undef, 0)) where {T1 <: Real}

    if !parametercheck(θ, model)
        error("Parameters not valid.")
    end

    y = fill(0, T + burnin)
    X = model.X

    p = length(model.pastObs)
    if p == 0
        P = 0
    else
        P = maximum(model.pastObs)
    end

    r = length(model.external)
    M = P

    if (size(X)[2] > 0) & (size(X)[2] != T)
        error("Number of regressors does not match T.")
    end

    logl = (model.link == "Log")
    lin = !logl

    nb = model.distr == "NegativeBinomial"
    gp = model.distr == "GPoisson"
    zi = model.zi

    β0 = θ.β0
    α = θ.α
    η = θ.η
    ϕ = θ.ϕ
    ϕ = θ.ϕ
    ω = θ.ω

    ν = fill(β0, T + burnin)
    λ = fill(ifelse(lin, β0, exp(β0)), T + burnin)

    if p == 0
        for i = 1:r
            ν[burnin + 1:end] .+= η[i]*X[i, :]
        end
        if logl
            λ = exp.(ν)
        end

        if nb
            y = rand.(NegativeBinomial.(ϕ[1], ϕ[1]./(ϕ[1] .+ λ)))
        elseif gp
            y = rand.(GPoisson.(λ, ϕ[1]))
        else
            y = rand.(Poisson.(λ))
        end

        y = y.*(rand(T + burnin) .>= ω)
        λzi = λ.*(1 - ω)

        return y[burnin + 1:end], ν[burnin + 1:end], λ[burnin + 1:end], λzi[burnin + 1:end]
    end

    νinit = β0/(1 - sum(α))
    λinit = ifelse(logl, exp(νinit), νinit)

    ν[1:M] = fill(νinit, M)
    λ[1:M] = fill(λinit, M)

    if nb
        y[1:M] = rand.(NegativeBinomial.(ϕ[1], ϕ[1]./(ϕ[1] .+ λ[1:M])))
    elseif gp
        y[1:M] = rand.(GPoisson.(λ[1:M], ϕ[1]))
    else
        y[1:M] = rand.(Poisson.(λ[1:M]))
    end

    if zi
        y[1:M] = y[1:M].*(rand(M) .>= ω)
    end

    for t = (M+1):(T + burnin)
        ν[t] = β0
        if logl
            ν[t] += dot(α, log.(y[t .- model.pastObs] .+ 1))
        else
            ν[t] += dot(α, y[t .- model.pastObs])
        end

        if r > 0
            ν[t] += dot(η, X[:, maximum([1, t - burnin])])
        end

        λ[t] = ν[t]

        if logl
            λ[t] = exp(λ[t])
        end

        if nb
            y[t] = rand(NegativeBinomial(ϕ[1], ϕ[1]/(ϕ[1] + λ[t])))
        elseif gp
            y[t] = rand(GPoisson(λ[t], ϕ[1]))
        else
            y[t] = rand(Poisson(λ[t]))
        end

        if zi
            y[t] = y[t]*(rand() >= ω)
        end

        if length(pinfirst) > 0
            if burnin + 1 <= t <= burnin + length(pinfirst)
                y[t] = pinfirst[t - burnin]
                λ[t] = pinfirst[t - burnin]
                ν[t] = ifelse(logl, log(pinfirst[t - burnin] + 1), pinfirst[t - burnin])
            end
        end
    end

    λzi = λ.*(1 - ω)

    return y[burnin + 1:end], ν[burnin + 1:end], λ[burnin + 1:end], λzi[burnin+1:end]
end

function simulate(T::Int64,
                  model::IIDModel,
                  θ::parameter;
                  burnin::Int64 = 500,
                  pinfirst::Array{T1, 1} = Vector{Float64}(undef, 0)) where {T1 <: Real}

    if !parametercheck(θ, model)
        error("Parameters not valid.")
    end

    y = fill(0, T + burnin)
    X = model.X
    r = length(model.external)

    if (size(X)[2] > 0) & (size(X)[2] != T)
        error("Number of regressors does not match T.")
    end

    logl = (model.link == "Log")
    lin = !logl

    nb = model.distr == "NegativeBinomial"
    gp = model.distr == "GPoisson"
    zi = model.zi

    β0 = θ.β0
    η = θ.η
    ϕ = θ.ϕ
    ϕ = θ.ϕ
    ω = θ.ω

    ν = fill(β0, T + burnin)
    λ = fill(ifelse(lin, β0, exp(β0)), T + burnin)

    for i = 1:r
        ν[burnin + 1:end] .+= η[i]*X[i, :]
    end
    if logl
        λ = exp.(ν)
    end

    if nb
        y = rand.(NegativeBinomial.(ϕ[1], ϕ[1]./(ϕ[1] .+ λ)))
    elseif gp
        y = rand.(GPoisson.(λ, ϕ[1]))
    else
        y = rand.(Poisson.(λ))
    end

    y = y.*(rand(T + burnin) .>= ω)
    λzi = λ.*(1 - ω)

    return y[burnin + 1:end], ν[burnin + 1:end], λ[burnin + 1:end], λzi[burnin + 1:end]
end

function simulate(T::Int64,
                  model::T1,
                  θ::parameter;
                  burnin::Int64 = 500,
                  pinfirst::Array{T2, 1} = Vector{Float64}(undef, 0)) where {T1 <: INARMA, T2 <: Real}
    if !parametercheck(θ, model)
        error("Parameters not valid.")
    end

    Y = fill(0, T + burnin)
    X = model.X

    p = length(model.pastObs)
    q = length(model.pastMean)
    if p == 0
        P = 0
    else
        P = maximum(model.pastObs)
    end
    if q == 0
        Q = 0
    else
        Q = maximum(model.pastMean)
    end

    r = length(model.external)

    if (size(X)[2] > 0) & (size(X)[2] != T)
        error("Number of regressors does not match T.")
    end

    M = maximum([P, Q])

    logl1 = (model.link[1] == "Log")
    logl2 = (model.link[2] == "Log")

    lin1 = !logl1
    lin2 = !logl2

    nb1 = model.distr[1] == "NegativeBinomial"
    nb2 = model.distr[2] == "NegativeBinomial"

    gp1 = model.distr[1] == "GPoisson"
    gp2 = model.distr[2] == "GPoisson"

    β0 = θ.β0
    α = θ.α
    β = θ.β
    η = θ.η
    ϕ = θ.ϕ
    ω = θ.ω

    rI = 0
    rE = 0
    iI = []
    iE = []
    if r > 0
        iE = findall(model.external)
        iI = setdiff(1:r, rE)
        rE = sum(iE)
        rI = r - rE
    end

    R = fill(0, T + burnin)
    Z = fill(0, T + burnin)

    μ = zeros(T + burnin)
    λ = fill(β0, T + burnin)

    if rI > 0
        for i = 1:rI
            λ[1:burnin] .+= fill(η[iI[i]]*X[iI[i], 1], burnin)
            λ[burnin + 1:end] .+= η[iI[i]].*X[iI[i], :]
        end
    end

    if logl1
        λ = exp.(λ)
    end

    if rE > 0
        for i = 1:rE
            μ[1:burnin] .+= fill(η[iE[i]]*X[iE[i], 1], burnin)
            μ[burnin + 1:end] .+= η[iE[i]].*X[iE[i], :]
        end
        if logl2
            μ = exp.(μ)
        end
    end

    if nb1
        p1 = ϕ[1]./(ϕ[1] .+ λ)
        R = rand.(NegativeBinomial.(ϕ[1], p1))
    elseif gp1
        R = rand.(GPoisson.(λ, ϕ[1]))
    else
        R = rand.(Poisson.(λ))
    end

    if model.zi
        R = R.*(rand(length(R)) .> ω)
    end

    if nb2
        p2 = ϕ[1 + nb1 + gp1]./(ϕ[1 + nb1 + gp1] .+ μ)
        Z = rand.(NegativeBinomial.(ϕ[1 + nb1 + gp1], p2))
    elseif gp2
        Z = rand.(GPoisson.(μ, ϕ[1 + gp1 + nb1]))
    else
        Z = rand.(Poisson.(μ))
    end

    Y[1:M] = R[1:M]

    pf = length(pinfirst)
    pfBool = pf > 0

    for t = M+1:T+burnin
        Y[t] = R[t] + Z[t]
        if p > 0
            Y[t] += sum(α.∘Y[t .- model.pastObs])
        end

        if q > 0
            Y[t] += sum(β.∘R[t .- model.pastMean])
        end

        if pfBool
            if 1 <= (t - burnin) <= pf
                Y[t] = pinfirst[t - burnin]
            end
        end
    end

    return Y[burnin + 1:end], R[burnin+1:end]
end

function simulate(T::Int64,
                  model::INARModel,
                  θ::parameter;
                  burnin::Int64 = 500,
                  pinfirst::Array{T1, 1} = Vector{Float64}(undef, 0)) where {T1 <: Real}
    if !parametercheck(θ, model)
        error("Parameters not valid.")
    end

    Y = fill(0, T + burnin)
    X = model.X

    p = length(model.pastObs)
    if p == 0
        P = 0
    else
        P = maximum(model.pastObs)
    end

    r = length(model.external)

    if (size(X)[2] > 0) & (size(X)[2] != T)
        error("Number of regressors does not match T.")
    end

    M = P

    logl1 = (model.link[1] == "Log")
    logl2 = (model.link[2] == "Log")

    lin1 = !logl1
    lin2 = !logl2

    nb1 = model.distr[1] == "NegativeBinomial"
    nb2 = model.distr[2] == "NegativeBinomial"

    gp1 = model.distr[1] == "GPoisson"
    gp2 = model.distr[2] == "GPoisson"

    β0 = θ.β0
    α = θ.α
    η = θ.η
    ϕ = θ.ϕ
    ω = θ.ω

    rI = 0
    rE = 0
    iI = []
    iE = []
    if r > 0
        iE = findall(model.external)
        iI = setdiff(1:r, rE)
        rE = sum(iE)
        rI = r - rE
    end

    R = fill(0, T + burnin)
    Z = fill(0, T+ burnin)

    μ = zeros(T + burnin)
    λ = fill(β0, T + burnin)

    if rI > 0
        for i = 1:rI
            λ[1:burnin] .+= fill(η[iI[i]]*X[iI[i], 1], burnin)
            λ[burnin + 1:end] .+= η[iI[i]].*X[iI[i], :]
        end
    end

    if logl1
        λ = exp.(λ)
    end

    if rE > 0
        for i = 1:rE
            μ[1:burnin] .+= fill(η[iE[i]]*X[iE[i], 1], burnin)
            μ[burnin + 1:end] .+= η[iE[i]].*X[iE[i], :]
        end
        if logl2
            μ = exp.(μ)
        end
    end

    if nb1
        p1 = ϕ[1]./(ϕ[1] .+ λ)
        R = rand.(NegativeBinomial.(ϕ[1], p1))
    elseif gp1
        R = rand.(GPoisson.(λ, ϕ[1]))
    else
        R = rand.(Poisson.(λ))
    end

    if model.zi
        R = R.*(rand(length(R)) .> ω)
    end

    if nb2
        p2 = ϕ[1 + nb1 + gp1]./(ϕ[1 + nb1 + gp1] .+ μ)
        Z = rand.(NegativeBinomial.(ϕ[1 + nb1 + gp1], p2))
    elseif gp2
        Z = rand.(GPoisson.(μ, ϕ[1 + nb1 + gp1]))
    else
        Z = rand.(Poisson.(μ))
    end

    Y[1:M] = R[1:M]

    pf = length(pinfirst)
    pfBool = pf > 0

    for t = M+1:T+burnin
        Y[t] = R[t] + Z[t]
        if p > 0
            Y[t] += sum(α.∘Y[t .- model.pastObs])
        end

        if pfBool
            if 1 <= (t - burnin) <= pf
                Y[t] = pinfirst[t - burnin]
            end
        end
    end

    return Y[burnin + 1:end], R[burnin+1:end]
end

function simulate(T::Int64,
                  model::INMAModel,
                  θ::parameter;
                  burnin::Int64 = 500,
                  pinfirst::Array{T1, 1} = Vector{Float64}(undef, 0)) where {T1 <: Real}
    if !parametercheck(θ, model)
        error("Parameters not valid.")
    end

    Y = fill(0, T + burnin)
    X = model.X

    q = length(model.pastMean)
    if q == 0
        Q = 0
    else
        Q = maximum(model.pastMean)
    end

    r = length(model.external)

    if (size(X)[2] > 0) & (size(X)[2] != T)
        error("Number of regressors does not match T.")
    end

    M = Q

    logl1 = (model.link[1] == "Log")
    logl2 = (model.link[2] == "Log")

    lin1 = !logl1
    lin2 = !logl2

    nb1 = model.distr[1] == "NegativeBinomial"
    nb2 = model.distr[2] == "NegativeBinomial"

    gp1 = model.distr[1] == "GPoisson"
    gp2 = model.distr[2] == "GPoisson"

    β0 = θ.β0
    β = θ.β
    η = θ.η
    ϕ = θ.ϕ
    ω = θ.ω

    rI = 0
    rE = 0
    iI = []
    iE = []
    if r > 0
        iE = findall(model.external)
        iI = setdiff(1:r, rE)
        rE = sum(iE)
        rI = r - rE
    end

    R = fill(0, T + burnin)
    Z = fill(0, T + burnin)

    μ = zeros(T + burnin)
    λ = fill(β0, T + burnin)

    if rI > 0
        for i = 1:rI
            λ[1:burnin] .+= fill(η[iI[i]]*X[iI[i], 1], burnin)
            λ[burnin + 1:end] .+= η[iI[i]].*X[iI[i], :]
        end
    end

    if logl1
        λ = exp.(λ)
    end

    if rE > 0
        for i = 1:rE
            μ[1:burnin] .+= fill(η[iE[i]]*X[iE[i], 1], burnin)
            μ[burnin + 1:end] .+= η[iE[i]].*X[iE[i], :]
        end
        if logl2
            μ = exp.(μ)
        end
    end

    if nb1
        p1 = ϕ[1]./(ϕ[1] .+ λ)
        R = rand.(NegativeBinomial.(ϕ[1], p1))
    elseif gp1
        R = rand.(GPoisson.(λ, ϕ[1]))
    else
        R = rand.(Poisson.(λ))
    end

    if model.zi
        R = R.*(rand(length(R)) .> ω)
    end

    if nb2
        p2 = ϕ[1 + nb1 + gp1]./(ϕ[1 + nb1 + gp1] .+ μ)
        Z = rand.(NegativeBinomial.(ϕ[1 + nb1 + gp1], p2))
    elseif gp2
        Z = rand.(GPoisson.(μ, ϕ[1 + nb1 + gp1]))
    else
        Z = rand.(Poisson.(μ))
    end

    Y[1:M] = R[1:M]

    pf = length(pinfirst)
    pfBool = pf > 0

    for t = M+1:T+burnin
        Y[t] = R[t] + Z[t]

        if q > 0
            Y[t] += sum(β.∘R[t .- model.pastMean])
        end

        if pfBool
            if 1 <= (t - burnin) <= pf
                Y[t] = pinfirst[t - burnin]
            end
        end
    end

    return Y[burnin + 1:end], R[burnin+1:end]
end

function simulate(T::Int64,
                  model::T1,
                  θ::Array{T2, 1};
                  burnin::Int64 = 500,
                  pinfirst::Array{T3, 1} = Vector{Float64}(undef, 0)) where {T1 <: CountModel, T2, T3 <: Real}
    if !parametercheck(θ, model)
        error("Parameters not valid.")
    end

    simulate(T, model, θ2par(θ, model), burnin = burnin, pinfirst = pinfirst)
end
