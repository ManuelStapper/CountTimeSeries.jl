# Compute residuals for INARMA models
# Similar to likelihood function, but returns E(Yₜ|Fₜ₋₁)

# INAR model

function fittedValues(y::Vector{Int64},
            model::INARModel,
            θ::parameter)::Vector{Float64}
    T = length(y)
    ymax = maximum(y)
    X = model.X

    r = length(model.external)

    logl1 = model.link[1] == "Log"
    lin1 = !logl1

    logl2 = model.link[2] == "Log"
    lin2 = !logl2

    nb1 = model.distr[1] == "NegativeBinomial"
    nb2 = model.distr[2] == "NegativeBinomial"
    if r == 0
        nb2 = false
    else
        if sum(model.external) == 0
            nb2 = false
        end
    end

    zi = model.zi

    p = length(model.pastObs)
    if p == 0
        P = 0
    else
        P = maximum(model.pastObs)
    end

    M = P

    if !CountTimeSeries.parametercheck(θ, model)
        return -Inf, fill(-Inf, T)
    end

    β0 = θ.β0
    αRaw = θ.α
    η = θ.η
    ϕ = θ.ϕ
    ω = θ.ω

    if p > 0
        if p < P
            α = zeros(P)
            α[model.pastObs] = αRaw
        else
            α = αRaw
        end
    end

    rE = sum(model.external)
    rI = sum(.!model.external)

    if r > 0
        iE = findall(model.external)
        iI = findall(.!model.external)

        ηI = η[iI]
        ηE = η[iE]

        λ, μ = CountTimeSeries.createλμ(T, β0, rI, iI, η, X, rE, iE, logl1, logl2)
    else
        λ, μ = CountTimeSeries.createλμ(T, β0, logl1)
    end
    
    out = λ .* (1 - ω) .+ μ
    for t = M+1:T
        out[t] = out[t] + sum(α .* y[t-1:-1:t-P])
    end
    out[1:M] = y[1:M]

    return out
end


# Approximation of residuals!

function fittedValues(y::Vector{Int64},
            model::INARMAModel,
            θ::parameter)::Vector{Float64}
    T = length(y)
    ymax = maximum(y)
    X = model.X
    r = Int64(length(model.external))

    logl1 = model.link[1] == "Log"
    lin1 = !logl1

    logl2 = model.link[2] == "Log"
    lin2 = !logl2

    nb1 = model.distr[1] == "NegativeBinomial"
    nb2 = model.distr[2] == "NegativeBinomial"
    if r == 0
        nb2 = false
    else
        if sum(model.external) == 0
            nb2 = false
        end
    end

    zi = model.zi

    p = length(model.pastObs)
    if p == 0
        P = 0
    else
        P = Int64(maximum(model.pastObs))
    end

    q = Int64(length(model.pastMean))
    if q == 0
        Q = 0
    else
        Q = Int64(maximum(model.pastMean))
    end
    M = maximum([P, Q])

    if !CountTimeSeries.parametercheck(θ, model)
        return fill(0.0, T)
    end

    β0 = Float64(θ.β0)
    αRaw = Vector{Float64}(θ.α)
    βRaw = Vector{Float64}(θ.β)
    η = Vector{Float64}(θ.η)
    ϕ = Vector{Float64}(θ.ϕ)
    ω = Float64(θ.ω)

    if p > 0
        if p < P
            α = zeros(P)
            α[model.pastObs] = αRaw
        else
            α = αRaw
        end
    end

    if q > 0
        if q < Q
            β = zeros(Q)
            β[model.pastMean] = βRaw
        else
            β = βRaw
        end
    end

    rE = sum(model.external)
    rI = r - rE

    if r > 0
        iE = findall(model.external)
        iI = findall(.!model.external)

        ηI = η[iI]
        ηE = η[iE]

        λ, μ = CountTimeSeries.createλμ(T, β0, rI, iI, η, X, rE, iE, logl1, logl2)
    else
        λ, μ = CountTimeSeries.createλμ(T, β0, logl1)
    end

    λ2 = λ .* (1 - ω)
    R2 = zeros(T)
    R2[1:M] = λ2[1:M]    

    for t = M+1:T
        R2[t] = y[t]*λ2[t] / (λ2[t] + sum(β .* λ2[t-1:-1:t-Q]) + sum(α .* y[t-1:-1:t-P]) + μ[t])
    end

    out = zeros(T)
    out[1:M] = y[1:M]

    for t = M+1:T
        out[t] = λ2[t] + sum(β .* R2[t-1:-1:t-Q]) + sum(α .* y[t-1:-1:t-P]) + μ[t]
    end

    return out
end