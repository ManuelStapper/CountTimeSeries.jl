# Wrapper function for prediction

"""
    predict(results, h, nChain, Xnew)
Function for forecasting Count Data models.

* `results`: Estimation results
* `h`: Number of steps to forecast
* `nChain`: Number of Chains for simulation based forecast (optional)
* `Xnew`: New values for regressors (only in case of regressors)

# Example
```julia-repl
# 10-step-ahead forecast
predict(results, 10, 10000)
```

The function either returns point forecasts if `nChain` is not specified or
generates multiple time series according to estiamtion results.
The latter is used to compute forecast intervals and is the default for INARMA models.
"""
function predict(results::INGARCHresults,
                 h::Int64,
                 Xnew::Array{T1, 2} = zeros(0, 0))  where {T1 <: Real}

    r = length(results.model.external)
    Xnew = Float64.(Xnew)

    if ndims(Xnew) == 1
        if r > 1
            error("Dimensions of Xnew and external do not match.")
        end

        if length(Xnew) != h
            error("Dimensions of Xnew and h do not match.")
        end

        Xnew = reshape(Xnew, (1, h))
    end

    if length(Xnew) != 0
        if (size(Xnew) == (h, r)) & (h != r)
            Xnew = convert(Array{Float64, 2}, Xnew')
        end

        if size(Xnew)[1] != r
            error("Dimensions of Xnew and external do not match.")
        end

        if size(Xnew)[2] != h
            error("Dimensions of Xnew and external do not match.")
        end
    end

    if ndims(Xnew) != 2
        error("Incorrect type of Xnew.")
    end

    y = results.y
    T = length(y)
    zi = results.model.zi
    nb = results.model.distr == "NegativeBinomial"
    gp = results.model.distr == "GPoisson"
    logl = results.model.link == "Log"
    lin = !logl
    pars = results.pars

    if typeof(results.model) != IIDModel
        p = length(results.model.pastObs)
    else
        p = 0
    end

    if typeof(results.model) == INGARCHModel
        q = length(results.model.pastMean)
    else
        q = 0
    end

    if p == 0
        P = 0
    else
        P = maximum(results.model.pastObs)
    end

    if q == 0
        Q = 0
    else
        Q = maximum(results.model.pastMean)
    end

    r = length(results.model.external)

    rI = 0
    rE = 0
    iI = []
    iE = []

    if r > 0
        iE = findall(results.model.external)
        iI = setdiff(1:r, rE)
        rE = sum(iE)
        rI = r - rE
    end

    β0 = pars.β0
    α = pars.α
    β = pars.β
    η = pars.η
    ϕ = pars.ϕ
    ω = pars.ω

    λ = zeros(T + h)
    λ[1:T] = results.λ

    if logl
        ν = log.(λ)
    else
        ν = λ
    end

    X = results.model.X
    if rE > 0
        for i = iE
            ν[1:T] = ν[1:T] .- η[i].*X[i, :]
        end
    end

    Y = zeros(T + h)
    Y[1:T] = y

    @inbounds for t = (T+1):(T+h)
        ν[t] = β0
        if p > 0
            if logl
                ν[t] += sum(α.*log.(Y[t .- results.model.pastObs] .+ 1))
            else
                ν[t] += sum(α.*Y[t .- results.model.pastObs])
            end
        end

        if q > 0
            ν[t] += sum(β.*ν[t .- results.model.pastMean])
        end

        if rI > 0
            ν[t] += sum(η[iI].*Xnew[iI, t - T])
        end

        λ[t] = copy(ν[t])

        if rE > 0
            λ[t] += sum(η[iE].*Xnew[iE, t - T])
        end

        if logl
            λ[t] = exp(λ[t])
        end

        Y[t] = copy(λ[t]*(1 - ω))
    end

    return Y[T+1:end]
end

function predict(results::INGARCHresults,
                 h::Int64,
                 nChain::Int64,
                 Xnew::Array{T1, 2} = zeros(0, 0)) where {T1 <: Real}
    r = length(results.model.external)
    Xnew = Float64.(Xnew)

    if ndims(Xnew) == 1
        if r > 1
            error("Dimensions of Xnew and external do not match.")
        end

        if length(Xnew) != h
            error("Dimensions of Xnew and h do not match.")
        end

        Xnew = reshape(Xnew, (1, h))
    end

    if length(Xnew) != 0
        if (size(Xnew) == (h, r)) & (h != r)
            Xnew = convert(Array{Float64, 2}, Xnew')
        end

        if size(Xnew)[1] != length(results.model.external)
            error("Dimensions of Xnew and external do not match.")
        end

        if size(Xnew)[2] != h
            error("Dimensions of Xnew and external do not match.")
        end
    end

    if ndims(Xnew) != 2
        error("Incorrect type of Xnew.")
    end

    y = results.y
    T = length(y)
    zi = results.model.zi
    nb = results.model.distr == "NegativeBinomial"
    gp = results.model.distr == "GPoisson"
    logl = results.model.link == "Log"
    lin = !logl
    pars = results.pars

    if typeof(results.model) != IIDModel
        p = length(results.model.pastObs)
    else
        p = 0
    end

    if typeof(results.model) == INGARCHModel
        q = length(results.model.pastMean)
    else
        q = 0
    end

    if p == 0
        P = 0
    else
        P = maximum(results.model.pastObs)
    end

    if q == 0
        Q = 0
    else
        Q = maximum(results.model.pastMean)
    end

    M = maximum([P, Q])

    λ = zeros(nChain, M + h)
    ν = zeros(nChain, M + h)

    rI = 0
    rE = 0
    iI = []
    iE = []

    if r > 0
        iE = findall(results.model.external)
        iI = setdiff(1:r, rE)
        rE = sum(iE)
        rI = r - rE
    end

    β0 = Float64(pars.β0)
    α = pars.α
    β = pars.β
    if nb | gp
        ϕ = pars.ϕ[1]
    else
        ϕ = 0.0
    end

    η = pars.η
    ω = pars.ω

    Y = zeros(Int64, nChain, M + h)
    λ = zeros(nChain, M + h)
    ν = zeros(nChain, M + h)

    λOld = results.λ

    if logl
        νOld = log.(λOld)
    else
        νOld = λOld
    end

    X = results.model.X
    if rE > 0
        for i = iE
            νOld = νOld .- η[i].*X[i, :]
        end
    end

    @inbounds for i = 1:nChain
        λ[i, 1:M] = λOld[(end - M + 1):end]
        ν[i, 1:M] = νOld[(end - M + 1):end]
        Y[i, 1:M] = y[(end - M + 1):end]
    end

    @simd for i = 1:nChain
        @inbounds for t = (M+1):(M+h)
            ν[i, t] = β0
            if p > 0
                if logl
                    ν[i, t] += sum(α.*log.(Y[i, t .- results.model.pastObs] .+ 1))
                else
                    ν[i, t] += sum(α.*Y[i, t .- results.model.pastObs])
                end
            end

            if q > 0
                ν[i, t] += sum(β.*ν[i, t .- results.model.pastMean])
            end

            if rI > 0
                ν[i, t] += sum(η[iI].*Xnew[iI, t - M])
            end

            λ[i, t] = ν[i, t]

            if rE > 0
                λ[i, t] += sum(η[iE].*Xnew[iE, t - M])
            end

            if logl
                λ[i, t] = exp(λ[i, t])
            end

            if nb
                p = ϕ/(ϕ + λ[i, t])
                Y[i, t] = rand(NegativeBinomial(ϕ, p))*(rand() > ω)
            elseif gp
                Y[i, t] = rand(GPoisson(λ[i, t], ϕ))*(rand() > ω)
            else
                Y[i, t] = rand(Poisson(λ[i, t]))*(rand() > ω)
            end
        end
    end
    Yout = Y[:, M+1:end]

    meanY = mean(Yout, dims = 1)[1, :]
    Q = zeros(2, h)

    for i = 1:h
        Q[:, i] = quantile(Yout[:, i], [0.025, 0.975])
    end

    return meanY, Q, Yout
end

function predict(results::INARMAresults,
                 h::Int64,
                 nChain::Int64,
                 Xnew::Array{T1, 2} = zeros(0, 0)) where {T1 <: Real}

    r = length(results.model.external)
    Xnew = Float64.(Xnew)

    if ndims(Xnew) == 1
        if length(results.model.external) > 1
            error("Dimensions of Xnew and external do not match.")
        end

        if length(Xnew) != h
            error("Dimensions of Xnew and h do not match.")
        end

        Xnew = reshape(Xnew, (1, h))
    end

    if length(Xnew) != 0
        if (size(Xnew) == (h, r)) & (h != r)
            Xnew = convert(Array{Float64, 2}, Xnew')
        end

        if size(Xnew)[1] != length(results.model.external)
            error("Dimensions of Xnew and external do not match.")
        end

        if size(Xnew)[2] != h
            error("Dimensions of Xnew and external do not match.")
        end
    else
        if length(results.model.external) > 0
            error("Dimensions of Xnew and external do not match.")
        end
    end

    if ndims(Xnew) != 2
        error("Incorrect type of Xnew.")
    end

    y = results.y
    T = length(y)
    zi = results.model.zi
    nb1 = results.model.distr[1] == "NegativeBinomial"
    nb2 = results.model.distr[2] == "NegativeBinomial"
    gp1 = results.model.distr[1] == "GPoisson"
    gp2 = results.model.distr[2] == "GPoisson"
    logl1 = results.model.link[1] == "Log"
    logl2 = results.model.link[2] == "Log"
    lin1 = !logl1
    lin2 = !logl2
    pars = results.pars

    if typeof(results.model) != INMAModel
        p = length(results.model.pastObs)
    else
        p = 0
    end

    if typeof(results.model) != INARModel
        q = length(results.model.pastMean)
    else
        q = 0
    end

    if p == 0
        P = 0
    else
        P = maximum(results.model.pastObs)
    end
    if q == 0
        Q = 0
    else
        Q = maximum(results.model.pastMean)
    end

    M = maximum([P, Q])

    rI = 0
    rE = 0
    iI = []
    iE = []

    if r > 0
        iE = findall(results.model.external)
        iI = setdiff(1:r, rE)
        rE = sum(iE)
        rI = r - rE
    end

    β0 = pars.β0
    α = pars.α
    β = pars.β
    if nb1 | gp1
        ϕ1 = pars.ϕ[1]
    end
    if nb2 | gp2
        ϕ2 = pars.ϕ[2]
    end
    η = Vector{Float64}(pars.η)
    ω = pars.ω

    Y = fill(0, (nChain, M + h))
    R = fill(0, (nChain, M + h))
    Z = fill(0, (nChain, M + h))

    μ = zeros(M + h)
    λ = fill(β0, M + h)

    # Initialization
    @inbounds for i = 1:M
        if rE > 0
            for ii = iE
                μ[i] += (η[ii]*results.model.X[ii, T - M + i])
                if results.model.link[2] == "Log"
                    μ[i] = exp(μ[i])
                end
            end
        end

        if rI > 0
            for ii = iI
                if results.model.link[2] == "Log"
                    λ[i] += exp(η[ii]*results.model.X[ii, T - M + i])
                else
                    λ[i] += η[ii]*results.model.X[ii, T - M + i]
                end
            end
        end

        if nb1
            dR = MixtureModel([NegativeBinomial(ϕ1, ϕ1/(ϕ1 + λ[i])), Poisson(0)], [1 - ω, ω])
        elseif gp1
            dR = MixtureModel([GPoisson(λ[i], ϕ1), Poisson(0)], [1 - ω, ω])
        else
            dR = MixtureModel([Poisson(λ[i]), Poisson(0)], [1 - ω, ω])
        end

        if nb2
            dZ = NegativeBinomial(ϕ2, ϕ2/(ϕ2 + μ[i]))
        elseif gp2
            dZ = GPoisson(μ[i], ϕ2)
        else
            dZ = Poisson(μ[i])
        end

        @inbounds for j = 1:nChain
            while true
                Rtemp = rand(dR)
                Ztemp = rand(dZ)
                if Rtemp + Ztemp <= y[T - M + i]
                    R[j, i] = Rtemp
                    Z[j, i] = Ztemp
                    break
                end
            end
            Y[j, i] = y[T - M + i]
        end
    end

    @inbounds for t = (M+1):(M+h)
        if rE > 0
            for i = iE
                if results.model.link[2] == "Log"
                    μ[t] += exp(η[i]*Xnew[i, t - M])
                else
                    μ[t] += η[i]*Xnew[i, t - M]
                end
            end
        end

        if rI > 0
            for i = iI
                if results.model.link[2] == "Log"
                    λ[t] += exp(η[i]*Xnew[i, t - M])
                else
                    λ[t] += η[i]*Xnew[i, t - M]
                end
                
            end
        end

        if nb1
            dR = MixtureModel([NegativeBinomial(ϕ1, ϕ1/(ϕ1 + λ[t])), Poisson(0)], [1 - ω, ω])
        elseif gp1
            dR = MixtureModel([GPoisson(λ[t], ϕ1), Poisson(0)], [1 - ω, ω])
        else
            dR = MixtureModel([Poisson(λ[t]), Poisson(0)], [1 - ω, ω])
        end

        if nb2
            dZ = NegativeBinomial(ϕ2, ϕ2/(ϕ2 + μ[t]))
        elseif gp2
            dZ = GPoisson(μ[t], ϕ2)
        else
            dZ = Poisson(μ[t])
        end

        @inbounds for i = 1:nChain
            R[i, t] = rand(dR)*(rand() >= ω)
            Z[i, t] = rand(dZ)
            Y[i, t] = R[i, t] + Z[i, t]

            for j = 1:p
                Y[i, t] += α[j]∘Y[i, t .- results.model.pastObs[j]]
            end

            for j = 1:q
                Y[i, t] += β[j]∘R[i, t .- results.model.pastMean[j]]
            end
        end
    end

    Y = Y[:, M+1:end]
    meanY = mean(Y, dims = 1)[1, :]
    Q = zeros(2, h)

    for i = 1:h
        Q[:, i] = quantile(Y[:, i], [0.025, 0.975])
    end

    return meanY, Q, Y
end

function predict(results::INARMAresults,
                 h::Int64,
                 Xnew::Array{T1, 2} = zeros(0, 0)) where {T1 <: Real}
    predict(results, h, 10000, Xnew)
end
