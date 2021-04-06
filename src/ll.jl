# Wrapper function for log likelihood

"""
    ll(y, model, θ; initiate = "first")
Likelihood function for Count Data models.

* `y`: Time series
* `model`: Model Specification
* `θ`: Parameters (Vector or parameter)
* `initiate`: How is the time series initiated?

The time series can be initiated by "first", the first observed value, "intercept",
the intercept parameter (possibly exponentiated if log-link), or "marginal", the marginal
mean of the process.

# Example
```julia-repl
ll(y, model, pars)
```
"""
function ll(y::Array{T, 1} where T<:Integer,
    model::INGARCHModel,
    θ::parameter;
    initiate = "first")::Tuple{Float64, Vector{Float64}}

    T = length(y)
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

    nb = model.distr == "NegativeBinomial"
    zi = model.zi

    M = convert(Int64, maximum([P, Q]))

    if !parametercheck(θ, model)
        return -Inf, fill(-Inf, T)
    end

    ϕ = θ.ϕ
    ω = θ.ω

    indrel = (M+1):T

    λ = LinPred(y, model, θ, initiate)
    lls = fill(0.0, T)

    if any(λ .< 0)
        return (-Inf, fill(-Inf, T))
    end

    if !nb
        d = Poisson.(λ[indrel])
        if ω > 0
            lls[indrel] = log.(pdf.(d, y[indrel]).*(1 - ω) .+ (y[indrel] .== 0).*ω)
        else
            lls[indrel] = logpdf.(d, y[indrel])
        end
        LL = sum(lls)
    else
        ptemp = @. ϕ[1]/(ϕ[1] + λ[indrel])
        d = NegativeBinomial.(ϕ[1], ptemp)
        if ω > 0
            lls[indrel] = log.(pdf.(d, y[indrel]).*(1 - ω) .+ (y[indrel].== 0).*ω)
        else
            lls[indrel] = logpdf.(d, y[indrel])
        end

        LL = sum(lls)
    end

    return LL, lls
end

function ll(y::Array{T, 1} where T<:Integer,
    model::INARCHModel,
    θ::parameter;
    initiate = "first")::Tuple{Float64, Vector{Float64}}

    T = length(y)
    X = model.X
    r = length(model.external)

    p = length(model.pastObs)
    if p == 0
        P = 0
    else
        P = maximum(model.pastObs)
    end

    M = P

    logl = (model.link == "Log")
    lin = !logl

    nb = model.distr == "NegativeBinomial"
    zi = model.zi

    M = P
    nObs = T - M
    nPar = 1 + p + r + nb + zi

    if !parametercheck(θ, model)
        return -Inf, fill(-Inf, T)
    end

    β0 = θ.β0
    αRaw = θ.α
    η = θ.η
    ϕ = θ.ϕ
    ω = θ.ω

    indrel = (M+1):T

    λ = LinPred(y, model, θ, initiate)
    if any(λ .< 0)
        return (-Inf, fill(-Inf, T))
    end

    lls = zeros(T)
    if !nb
        d = Poisson.(λ[indrel])
        if length(ω) == 1
            lls[indrel] = log.(pdf.(d, y[indrel]).*(1 - ω) .+ (y[indrel].== 0).*ω)
        elseif length(ω) == 0
            lls[indrel] = logpdf.(d, y[indrel])
        end
        LL = sum(lls)
    else
        ptemp = ϕ[1]./(ϕ[1] .+ λ[indrel])
        d = NegativeBinomial.(ϕ[1], ptemp)
        if length(ω) == 1
            lls[indrel] = log.(pdf.(d, y[indrel]).*(1 - ω) .+ (y[indrel].== 0).*ω)
        elseif length(ω) == 0
            lls[indrel] = logpdf.(d, y[indrel])
        end

        LL = sum(lls)
    end

    return LL, lls
end

function ll(y::Array{T, 1} where T<:Integer,
    model::IIDModel,
    θ::parameter;
    initiate = "first")::Tuple{Float64, Vector{Float64}}

    T = length(y)
    X = model.X

    r = length(model.external)

    logl = (model.link == "Log")
    lin = !logl

    nb = model.distr == "NegativeBinomial"
    zi = model.zi

    nObs = T
    nPar = 1 + r + nb + zi

    if !parametercheck(θ, model)
        return -Inf, fill(-Inf, T)
    end

    β0 = θ.β0
    η = θ.η
    ϕ = θ.ϕ
    ω = θ.ω

    λ = LinPred(y, model, θ, initiate)
    if any(λ .< 0)
        return (-Inf, fill(-Inf, T))
    end

    if !nb
        d = Poisson.(λ)
        if length(ω) == 1
            lls = log.(pdf.(d, y).*(1 - ω) .+ (y.== 0).*ω)
        elseif length(ω) == 0
            lls = logpdf.(d, y)
        end
        LL = sum(lls)
    else
        ptemp = ϕ[1]./(ϕ[1] .+ λ)
        d = NegativeBinomial.(ϕ[1], ptemp)
        if length(ω) == 1
            lls = log.(pdf.(d, y).*(1 - ω) .+ (y.== 0).*ω)
        elseif length(ω) == 0
            lls = logpdf.(d, y)
        end

        LL = sum(lls)
    end

    return LL, lls
end

function ll(y::Array{T, 1} where T<:Integer,
    model::INMAModel,
    θ::parameter)::Tuple{Float64, Vector{Float64}}

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

    q = length(model.pastMean)
    if q == 0
        Q = 0
    else
        Q = maximum(model.pastMean)
    end
    M = Q

    if !parametercheck(θ, model)
        return -Inf, fill(-Inf, T)
    end

    β0 = θ.β0
    βRaw = θ.β
    η = θ.η
    ϕ = θ.ϕ
    ω = θ.ω

    if q > 0
        if q < Q
            β = zeros(Q)
            β[model.pastMean] = βRaw
        else
            β = βRaw
        end
    end

    rE = sum(model.external)
    rI = sum(.!model.external)

    if r > 0
        iE = findall(model.external)
        iI = findall(.!model.external)

        ηI = η[iI]
        ηE = η[iE]
    end

    μ = zeros(T)
    λ = fill(β0, T)

    if rI > 0
        for i = 1:rI
            λ .+= η[iI[i]].*X[iI[i], :]
        end
    end

    if logl1
        λ = exp.(λ)
    end

    if rE > 0
        for i = 1:rE
            μ .+= η[iE[i]].*X[iE[i], :]
        end
        if logl2
            μ = exp.(μ)
        end
    end

    PR = zeros(ymax + 1, T)
    PZ = zeros(ymax + 1, T)
    for t = 1:T
        if nb1
            PR[:, t] = pdf.(NegativeBinomial(ϕ[1], ϕ[1]/(ϕ[1] + λ[t])), 0:ymax)
        else
            PR[:, t] = pdf.(Poisson(λ[t]), 0:ymax)
        end

        if zi
            PR[2:ymax + 1, t] = PR[2:ymax + 1, t].*(1 .- ω./(1 .- PR[1, t]))
            PR[1, t] += ω
        end

        if rE > 0
            if nb2
                PZ[:, t] = pdf.(NegativeBinomial(ϕ[2], ϕ[2]/(ϕ[2] + μ[t])), 0:ymax)
            else
                PZ[:, t] = pdf.(Poisson(μ[t]), 0:ymax)
            end
        else
            PZ[:, t] = [1; fill(0, ymax)]
        end
    end

    lls = zeros(T)

    if q == 0
        for t = M+1:T
            lls[t] = log(dot(PR[1:y[t]+1], PZ[y[t]+1:-1:1]))
        end
        LL = sum(lls[M+1:end])

        return LL, lls
    end

    # Compute P(Σ α_i∘Y_{t-i} + Z_t = k) first --> pARZ
    PARZ = PZ

    if Q == 1
        QQ = zeros(ymax + 1, ymax + 1, T)
        a = zeros(ymax + 1, T)
        ϕmat = zeros(ymax + 1, T)
        u = zeros(ymax + 1, T)
        lls = zeros(T)
        B = zeros(ymax + 1, ymax + 1)

        for k = 0:ymax
            B[:, k + 1] = pdf.(Binomial(k, β[1]), 0:ymax)
        end

        a[1:y[M] + 1, M] = PR[1:y[M] + 1, M]./sum(PR[1:y[M] + 1, M])
        lls[M] = sum(a[:, M])
        ϕmat[:, M] = a[:, M]./lls[M]
        for t = M+1:T
            for k = 0:y[t]
                temp = PARZ[1:y[t] - k + 1, t]
                for l = 0:y[t-1]
                    QQ[k + 1, l+1, t] = dot(temp, B[y[t] - k + 1:-1:1, l+1])
                end
            end

            u[:, t] = diagm(PR[:, t])*QQ[:, :, t]*ϕmat[:, t-1]
            lls[t] = sum(u[:, t])
            ϕmat[:, t] = u[:, t]./lls[t]
        end

        lls[M:end] = log.(lls[M:end])
        LL = sum(lls[M+1:end])

        return LL, lls
    end

    if Q == 2
        b = zeros(T, ymax + 1, ymax + 1, ymax + 1)
        w = zeros(T)

        B1 = zeros(ymax + 1, ymax + 1)
        B2 = zeros(ymax + 1, ymax + 1)

        for i = 0:ymax
            B1[:, i + 1] = pdf.(Binomial(i, β[1]), 0:ymax)
            B2[:, i + 1] = pdf.(Binomial(i, β[2]), 0:ymax)
        end
        B = zeros(ymax + 1, ymax + 1, ymax + 1)
        for i = 0:ymax, j = 0:ymax
            B[i + 1, j + 1, :] = convolution(B1[i + 1, :], B2[j + 1, :])
        end

        for r1 = 0:y[M]
            temp1 = convolution(B1[:, r1 + 1], PARZ[:, M+1])
            trunc1 = PR[r1 + 1, M]/sum(PR[1:y[M]+1, M])
            for r2 = 0:y[M-1]
                temp = convolution(temp1, B2[:, r2 + 1])
                trunc2 = PR[r2 + 1, M - 1]/sum(PR[1:y[M-1]+1, M - 1])
                b[M+1, 1:y[M+1]+1, r1 + 1, r2 + 1] = PR[1:y[M+1]+1, M+1] .* trunc1 .* trunc2 .*temp[y[M+1]+1:-1:1]
            end
        end

        w[M+1] = sum(b[M+1, :, :, :])
        b[M+1, :, :, :] = b[M+1, :, :, :]./w[M+1]

        for t = M+2:T
            @inbounds for r1 = 0:ymax
                temp1 = convolution(B1[:, r1 + 1], PARZ[:, t])

                for r2 = 0:ymax
                    temp = convolution(temp1, B2[:, r2 + 1], y[t])
                    b[t, 1:y[t] + 1, r1 + 1, r2 + 1] = PR[1:y[t]+1, t].*sum(b[t-1, r1 + 1, r2 + 1, 1:y[t-3] + 1]).*temp[y[t]+1:-1:1]
                end
            end
            w[t] = sum(b[t, :, :, :])
            b[t, :, :, :] = b[t, :, :, :]./w[t]
        end

        lls = [fill(0, M); log.(w[M+1:end])]
        LL = sum(lls)
        bFinal = b[T, :, :, :]
    end

    if Q >= 3
        error("Exact likelihood evaluation for INARMA(p, q) with q ≥ 3 not supported.")
    end

    return LL, lls
end

function ll(y::Array{T, 1} where T<:Integer,
    model::T where T<:INGARCH,
    θ::Array{T, 1} where T<: AbstractFloat;
    initiate = "first")::Tuple{Float64, Vector{Float64}}

    θNew = θ2par(θ, model)
    return ll(y, model, θNew, initiate = initiate)
end

function ll(y::Array{T, 1} where T<:Integer,
    model::T where T<:INARMA,
    θ::Array{T, 1} where T<: AbstractFloat)::Tuple{Float64, Vector{Float64}}

    θNew = θ2par(θ, model)
    return ll(y, model, θNew)
end


####################################
### Ab hier wird alles schneller ###
####################################

function Pmat(λ::Vector{Float64},
    μ::Vector{Float64},
    ymax::Int64,
    T::Int64,
    nb1::Bool,
    nb2::Bool,
    ϕ::Vector{Float64},
    zi::Bool,
    ω::Float64,
    rE::Int64)::Tuple{Matrix{Float64}, Matrix{Float64}}

    PR = zeros(ymax + 1, T)
    PZ = zeros(ymax + 1, T)

    if nb1 & !zi
        for t = 1:T
            PR[:, t] = pdf.(NegativeBinomial(ϕ[1], ϕ[1]/(ϕ[1] + λ[t])), 0:ymax)
        end
    end
    if !nb1 & !zi
        for t = 1:T
            PR[:, t] = pdf.(Poisson(λ[t]), 0:ymax)
        end
    end

    if nb1 & zi
        PRtemp = zeros(T)
        for t = 1:T
            PRtemp = pdf.(NegativeBinomial(ϕ[1], ϕ[1]/(ϕ[1] + λ[t])), 0:ymax)
            if ω + PRtemp[1] < 1
                PR[2:ymax + 1, t] = PRtemp[2:ymax + 1].*(1 .- ω./(1 .- PRtemp[1]))
                PR[1, t] = PRtemp[1] + ω
            else
                PR[2:ymax + 1, t] .= 0.0
                PR[1, t] = 1.0
            end

        end
    end

    if !nb1 & zi
        PRtemp = zeros(T)
        for t = 1:T
            PRtemp = pdf.(Poisson(λ[t]), 0:ymax)
            if ω + PRtemp[1] < 1
                PR[2:ymax + 1, t] = PRtemp[2:ymax + 1].*(1 .- ω./(1 .- PRtemp[1]))
                PR[1, t] = PRtemp[1] + ω
            else
                PR[2:ymax + 1, t] .= 0.0
                PR[1, t] = 1.0
            end
        end
    end

    if nb2 & (rE > 0)
        for t = 1:T
            PZ[:, t] = pdf.(NegativeBinomial(ϕ[2], ϕ[2]/(ϕ[2] + μ[t])), 0:ymax)
        end
    end

    if !nb2 & (rE > 0)
        for t = 1:T
            PZ[:, t] = pdf.(Poisson(μ[t]), 0:ymax)
        end
    end

    if rE == 0
        PZ[1, :] .= 1.0
    end

    return PR, PZ
end

function Q1funInner(y::Vector{Int64},
                    t::Int64,
                    ymax::Int64,
                    PARZt::Vector{Float64},
                    PRt::Vector{Float64},
                    B::Matrix{Float64},
                    ϕmatOld::Vector{Float64})::Vector{Float64}
    #
    QQ = zeros(ymax + 1, ymax + 1)
    @inbounds for k = 0:y[t]
        temp = PARZt[y[t] - k + 1:-1:1]
        QQ[k+1, 1:y[t-1]+1] = temp'B[1:y[t] - k + 1, 1:y[t-1]+1]
    end

    # return diagm(PRt)*QQ*ϕmatOld # u
    return PRt.*(QQ*ϕmatOld) # u
end

function Q1fun(y::Vector{Int64},
               ymax::Int64,
               T::Int64,
               β::Vector{Float64},
               PR::Matrix{Float64},
               M::Int64,
               PARZ::Matrix{Float64})::Tuple{Float64, Vector{Float64}}
    a = zeros(ymax + 1, T)
    lls = zeros(T)
    B = zeros(ymax + 1, ymax + 1)

    for k = 0:ymax
        B[:, k + 1] = pdf.(Binomial(k, β[1]), 0:ymax)
    end

    a[1:y[M] + 1, M] = PR[1:y[M] + 1, M]./sum(PR[1:y[M] + 1, M])
    lls[M] = sum(a[:, M])
    ϕmatOld = a[:, M]./lls[M]

    @inbounds for t = M+1:T
        u = Q1funInner(y, t, ymax, PARZ[:, t], PR[:, t], B, ϕmatOld)
        lls[t] = sum(u)
        ϕmatOld = u./lls[t]
    end

    lls[M:end] = log.(lls[M:end])
    LL = sum(lls[M+1:end])

    return LL, lls
end

function createB(ymax::Int64, β::Vector{Float64})::Tuple{Matrix{Float64}, Matrix{Float64}}
    B1 = zeros(ymax + 1, ymax + 1)
    B2 = zeros(ymax + 1, ymax + 1)

    for i = 0:ymax
        B1[:, i + 1] = pdf.(Binomial(i, β[1]), 0:ymax)
        B2[:, i + 1] = pdf.(Binomial(i, β[2]), 0:ymax)
    end
    B1, B2
end

function Q2funDeeper(t::Int64,
    y::Vector{Int64},
    ymax::Int64,
    PARZt::Vector{Float64},
    PRt::Vector{Float64},
    B1::Matrix{Float64},
    B2::Matrix{Float64},
    bOldSum::Matrix{Float64})::Tuple{Float64, Array{Float64, 2}}

    bNew = zeros(y[t-2] + 1, ymax + 1, y[t] + 1)

    bOldSumRel = bOldSum[1:y[t-2] + 1, :]
    B2rel = B2[:, 1:y[t-2]+1]
    @simd for r1 = 0:y[t-1]
        # Convolution of β_1 ∘ R_{t-1}, AR part and regressors Z_t
        temp1 = convolution(B1[:, r1 + 1], PARZt)
        tempMat = convolutionRevScale(temp1, B2rel, y[t], PRt)
        bNew[:, r1 + 1, :] = bOldSumRel[:, r1 + 1].*tempMat
    end

    wt = sum(bNew)

    return wt, sum(bNew ./ wt, dims = 1)[1, :, :]
end

function Q2funInner(y::Vector{Int64},
                    ymax::Int64,
                    M::Int64,
                    T::Int64,
                    B1::Matrix{Float64},
                    B2::Matrix{Float64},
                    PARZ::Matrix{Float64},
                    PR::Matrix{Float64})::Vector{Float64}
    bOld = zeros(ymax + 1, ymax + 1, ymax + 1)
    w = zeros(T)

    for r1 = 0:y[M]
        temp1 = convolution(B1[:, r1 + 1], PARZ[:, M+1])
        trunc1 = PR[r1 + 1, M]/sum(PR[1:y[M]+1, M])
        for r2 = 0:y[M-1]
            temp = convolution(temp1, B2[:, r2 + 1])
            trunc2 = PR[r2 + 1, M - 1]/sum(PR[1:y[M-1]+1, M - 1])
            bOld[r2 + 1, r1 + 1, 1:y[M+1]+1] = PR[1:y[M+1]+1, M+1] .* trunc1 .* trunc2 .*temp[y[M+1]+1:-1:1]
        end
    end

    w[M+1] = sum(bOld)
    bOld = bOld./w[M+1]
    bOldSum = sum(bOld[1:y[M-1] + 1, :, :], dims = 1)[1, :, :]

    @inbounds for t = M+2:T
        PARZt = PARZ[:, t]
        PRt = PR[:, t]
        w[t], bOldSum = Q2funDeeper(t, y, ymax, PARZt, PRt[1:y[t] + 1], B1, B2, bOldSum)
    end

    return w
end

function Q2fun(y::Vector{Int64},
               ymax::Int64,
               T::Int64,
               β::Vector{Float64},
               PARZ::Matrix{Float64},
               PR::Matrix{Float64},
               M::Int64)::Tuple{Float64, Vector{Float64}}

    B1, B2 = createB(ymax, β)

    w = Q2funInner(y, ymax, M, T, B1, B2, PARZ, PR)

    lls = [fill(0, M); log.(w[M+1:end])]
    LL = sum(lls)
    return LL, lls
end



function ll(y::Array{T, 1} where T<:Integer,
    model::INARMAModel,
    θ::parameter)::Tuple{Float64, Vector{Float64}}

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

    if !parametercheck(θ, model)
        return -Inf, fill(-Inf, T)
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

        λ, μ = createλμ(T, β0, rI, iI, η, X, rE, iE, logl1, logl2)
    else
        λ, μ = createλμ(T, β0, logl1)
    end

    PR, PZ = Pmat(λ, μ, ymax, T, nb1, nb2, ϕ, zi, ω, rE)

    lls = zeros(T)

    # Below only for INAR Process, rather irrelevant
    if q == 0
        PY = zeros(p, ymax + 1, ymax + 1)
        for i = 1:p
            for j = sort(unique(y))
                PY[i, j + 1, 1:j + 1] = pdf.(Binomial.(j, α[i]), 0:j)
            end
        end

        PAR = zeros(ymax + 1, P + 1)

        for t = M+1:T
            for i = 1:P
                PAR[:, i] = PY[i, y[t-i] + 1, :]
            end
            PAR[:, P + 1] = PZ[:, t]

            ptemp = GetProbFromP(PAR)
            lls[t] = log(dot(PR[1:y[t]+1], ptemp[y[t]+1:-1:1]))
        end
        LL = sum(lls[M+1:end])

        return LL, lls
    end

    # Compute P(Σ α_i∘Y_{t-i} + Z_t = k) first --> pARZ
    PAR = zeros(ymax + 1, P + 1)
    PARZ = zeros(ymax + 1, T)
    for t = M+1:T
        for i = 1:P
            PAR[:, i] = pdf.(Binomial(y[t - i], α[i]), 0:ymax)
        end
        PAR[:, end] = PZ[:, t]
        PARZ[:, t] = GetProbFromP(PAR)
    end

    if Q == 1
        return Q1fun(y, ymax, T, β, PR, M, PARZ)
    end

    if Q == 2
        return Q2fun(y, ymax, T, β, PARZ, PR, M)
    end

    if Q >= 3
        error("Exact likelihood evaluation for INARMA(p, q) with q ≥ 3 not supported.")
    end

    return LL, lls
end

function CoreINAR(y::Vector{Int64},
                  ymax::Int64,
                  P::Int64,
                  M::Int64,
                  T::Int64,
                  PY::Array{Float64, 3},
                  PZ::Matrix{Float64},
                  PR::Matrix{Float64})::Vector{Float64}
    #
    lls = zeros(T)
    PAR = zeros(ymax + 1, P + 1)
    @inbounds for t = M+1:T
        out = 0.0
        for i = 1:P
            PAR[:, i] = PY[i, y[t-i] + 1, :]
        end
        PAR[:, P + 1] = PZ[:, t]

        ptemp = GetProbFromP(PAR)
        for i = 0:y[t]
            out += PR[i+1]*ptemp[y[t] - i + 1]
        end
        lls[t] = log(out)
    end
    lls
end

function CreatePY(y::Vector{Int64},
    ymax::Int64,
    p::Int64,
    α::Vector{Float64})::Array{Float64, 3}

    PY = zeros(p, ymax + 1, ymax + 1)
    allVals = sort(unique(y))
    @inbounds for i = 1:p
        for j = allVals
            PY[i, j + 1, 1:j + 1] = pdf.(Binomial.(j, α[i]), 0:j)
        end
    end
    PY
end

function createλμ(T::Int64,
    β0::Float64,
    rI::Int64,
    iI::Vector{Int64},
    η::Vector{Float64},
    X::Matrix{Float64},
    rE::Int64,
    iE::Vector{Int64},
    logl1::Bool,
    logl2::Bool)::Tuple{Vector{Float64}, Vector{Float64}}

    μ = zeros(T)
    λ = fill(β0, T)

    if rI > 0
        λ .+= (η[iI]'X[iI, :])[1, :]
    end

    if logl1
        λ = exp.(λ)
    end

    if rE > 0
        μ .+= (η[iE]'X[iE, :])[1, :]

        if logl2
            μ = exp.(μ)
        end
    end
    λ, μ
end

function createλμ(T::Int64,
    β0::Float64,
    logl1::Bool)::Tuple{Vector{Float64}, Vector{Float64}}

    if logl1
        λ = fill(exp(β0), T)
    else
        λ = fill(β0, T)
    end

    λ, zeros(T)
end

function ll(y::Array{T, 1} where T<:Integer,
    model::INARModel,
    θ::parameter)::Tuple{Float64, Vector{Float64}}

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

    if !parametercheck(θ, model)
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

        λ, μ = createλμ(T, β0, rI, iI, η, X, rE, iE, logl1, logl2)
    else
        λ, μ = createλμ(T, β0, logl1)
    end
    PR, PZ = Pmat(λ, μ, ymax, T, nb1, nb2, ϕ, zi, ω, rE)
    PY = CreatePY(y, ymax, p, α)
    lls = CoreINAR(y, ymax, P, M, T, PY, PZ, PR)

    LL = sum(lls[M+1:end])

    return LL, lls
end
