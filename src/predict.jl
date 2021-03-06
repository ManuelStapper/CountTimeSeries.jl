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
    h::T where T<:Integer,
    Xnew::Array{T, 2} where T<:AbstractFloat = zeros(0, 0))

    r = length(results.model.external)

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

    ??0 = pars.??0
    ?? = pars.??
    ?? = pars.??
    ?? = pars.??
    ?? = pars.??
    ?? = pars.??

    ?? = zeros(T + h)
    ??[1:T] = results.??

    if logl
        ?? = log.(??)
    else
        ?? = ??
    end

    X = results.model.X
    if rE > 0
        for i = iE
            ??[1:T] = ??[1:T] .- ??[i].*X[i, :]
        end
    end

    Y = zeros(T + h)
    Y[1:T] = y

    for t = (T+1):(T+h)
        ??[t] = ??0
        if p > 0
            if logl
                ??[t] += sum(??.*log.(Y[t .- results.model.pastObs] .+ 1))
            else
                ??[t] += sum(??.*Y[t .- results.model.pastObs])
            end
        end

        if q > 0
            ??[t] += sum(??.*??[t .- results.model.pastMean])
        end

        if rI > 0
            ??[t] += sum(??[iI].*Xnew[iI, t - T])
        end

        ??[t] = copy(??[t])

        if rE > 0
            ??[t] += sum(??[iE].*Xnew[iE, t - T])
        end

        if logl
            ??[t] = exp(??[t])
        end

        Y[t] = copy(??[t]*(1 - ??))
    end

    return Y[T+1:end]
end

function predict(results::INGARCHresults,
    h::T where T<:Integer,
    nChain::T where T<:Integer,
    Xnew::Array{T, 2} where T<:AbstractFloat = zeros(0, 0))

    r = length(results.model.external)

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

    ?? = zeros(nChain, M + h)
    ?? = zeros(nChain, M + h)

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

    ??0 = Float64(pars.??0)
    ?? = pars.??
    ?? = pars.??
    if nb
        ?? = pars.??[1]
    else
        ?? = 0.0
    end

    ?? = pars.??
    ?? = pars.??

    Y = zeros(Int64, nChain, M + h)
    ?? = zeros(nChain, M + h)
    ?? = zeros(nChain, M + h)

    ??Old = results.??

    if logl
        ??Old = log.(??Old)
    else
        ??Old = ??Old
    end

    X = results.model.X
    if rE > 0
        for i = iE
            ??Old = ??Old .- ??[i].*X[i, :]
        end
    end

    for i = 1:nChain
        ??[i, 1:M] = ??Old[(end - M + 1):end]
        ??[i, 1:M] = ??Old[(end - M + 1):end]
        Y[i, 1:M] = y[(end - M + 1):end]
    end

    @simd for i = 1:nChain
        for t = (M+1):(M+h)
            ??[i, t] = ??0
            if p > 0
                if logl
                    ??[i, t] += sum(??.*log.(Y[i, t .- results.model.pastObs] .+ 1))
                else
                    ??[i, t] += sum(??.*Y[i, t .- results.model.pastObs])
                end
            end

            if q > 0
                ??[i, t] += sum(??.*??[i, t .- results.model.pastMean])
            end

            if rI > 0
                ??[i, t] += sum(??[iI].*Xnew[iI, t - M])
            end

            ??[i, t] = ??[i, t]

            if rE > 0
                ??[i, t] += sum(??[iE].*Xnew[iE, t - M])
            end

            if logl
                ??[i, t] = exp(??[i, t])
            end

            if nb
                p = ??/(?? + ??[i, t])
                Y[i, t] = rand(NegativeBinomial(??, p))*(rand() > ??)
            else
                Y[i, t] = rand(Poisson(??[i, t]))*(rand() > ??)
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
    h::T where T<:Integer,
    nChain::T where T<:Integer,
    Xnew::Array{T, 2} where T<:AbstractFloat = zeros(0, 0))

    r = length(results.model.external)

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

    ??0 = pars.??0
    ?? = pars.??
    ?? = pars.??
    if nb1
        ??1 = pars.??[1]
    end
    if nb2
        ??2 = pars.??[2]
    end
    ?? = Vector{Float64}(pars.??)
    ?? = pars.??

    Y = fill(0, (nChain, M + h))
    R = fill(0, (nChain, M + h))
    Z = fill(0, (nChain, M + h))

    ?? = zeros(M + h)
    ?? = fill(??0, M + h)

    # Initialization
    for i = 1:M
        if rE > 0
            for i = iE
                ??[i] += (??[i]*results.model.X[i, T - M + i])
            end
        end

        if rI > 0
            for i = iI
                ??[i] += ??[i]*results.model.X[i, T - M + i]
            end
        end

        if nb1
            dR = MixtureModel([NegativeBinomial(??1, ??1/(??1 + ??[i])), Poisson(0)], [1 - ??, ??])
        else
            dR = MixtureModel([Poisson(??[i]), Poisson(0)], [1 - ??, ??])
        end

        if nb2
            dZ = NegativeBinomial(??2, ??2/(??2 + ??[i]))
        else
            dZ = Poisson(??[i])
        end

        for j = 1:nChain
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

    for t = (M+1):(M+h)
        if rE > 0
            for i = iE
                ??[t] += ??[i]*Xnew[i, t - M]
            end
        end

        if rI > 0
            for i = iI
                ??[t] += ??[i]*Xnew[i, t - M]
            end
        end

        if nb1
            dR = MixtureModel([NegativeBinomial(??1, ??1/(??1 + ??[t])), Poisson(0)], [1 - ??, ??])
        else
            dR = MixtureModel([Poisson(??[t]), Poisson(0)], [1 - ??, ??])
        end

        if nb2
            dZ = NegativeBinomial(??2, ??2/(??2 + ??[t]))
        else
            dZ = Poisson(??[t])
        end

        for i = 1:nChain
            R[i, t] = rand(dR)*(rand() >= ??)
            Z[i, t] = rand(dZ)
            Y[i, t] = R[i, t] + Z[i, t]

            for j = 1:p
                Y[i, t] += ??[j]???Y[i, t .- results.model.pastObs[j]]
            end

            for j = 1:q
                Y[i, t] += ??[j]???R[i, t .- results.model.pastMean[j]]
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
    h::T where T<:Integer,
    Xnew::Array{T, 2} where T<:AbstractFloat = zeros(0, 0))

    predict(results, h, 10000, Xnew)
end
