# Function to translate vector of parameters to struct

"""
    θ2par(θ, model)
Function two convert parameter vector to struct

* `θ`: Parameter vector
* `model`: Model specification CountModel, INGARCHModel, INARMAModel, ...

Parameter vector needs to be of suitable length.

# Example
```julia-repl
model = Model(pastObs = 1) # Specify INARCH(1)
θ2par([10, 0.5], model)
```
"""
function θ2par(θ::Vector{T}, model::INGARCHModel)::parameter where {T <: Real}
    θ = Float64.(θ)

    p = length(model.pastObs)
    q = length(model.pastMean)
    r = length(model.external)
    nb = model.distr == "NegativeBinomial"
    zi = model.zi

    nPar = 1 + p + q + r + nb + zi

    if length(θ) != nPar
        error("Length of θ does not match model specification.")
    end

    β0 = θ[1]
    used = 1
    α = θ[used + 1:used + p]
    used += p
    β = θ[used + 1:used + q]
    used += q
    η = θ[used + 1:used + r]
    used += r
    ϕ = θ[used + 1:used + nb]
    if length(ϕ) == 0
        ϕ = Vector{Float64}([])
    end
    if zi
        ω = θ[end]
    else
        ω = 0.0
    end

    parameter(β0, α, β, η, ϕ, ω)
end


function θ2par(θ::Vector{T}, model::INARCHModel)::parameter where {T <: Real}
    θ = Float64.(θ)
    p = length(model.pastObs)
    r = length(model.external)
    nb = model.distr == "NegativeBinomial"
    zi = model.zi

    nPar = 1 + p + r + nb + zi

    if length(θ) != nPar
        error("Length of θ does not match model specification.")
    end

    β0 = θ[1]
    used = 1
    α = θ[used + 1:used + p]
    used += p
    η = θ[used + 1:used + r]
    used += r
    ϕ = θ[used + 1:used + nb]
    if length(ϕ) == 0
        ϕ = Vector{Float64}([])
    end
    if zi
        ω = θ[end]
    else
        ω = 0.0
    end

    parameter(β0, α, Vector{Float64}([]), η, ϕ, ω)
end

function θ2par(θ::Vector{T}, model::IIDModel)::parameter where {T <: Real}
    θ = Float64.(θ)
    r = length(model.external)
    nb = model.distr == "NegativeBinomial"
    zi = model.zi

    nPar = 1 + r + nb + zi

    if length(θ) != nPar
        error("Length of θ does not match model specification.")
    end

    β0 = θ[1]
    used = 1
    η = θ[used + 1:used + r]
    used += r
    ϕ = θ[used + 1:used + nb]
    if length(ϕ) == 0
        ϕ = Vector{Float64}([])
    end
    if zi
        ω = θ[end]
    else
        ω = 0.0
    end

    parameter(β0, Vector{Float64}([]), Vector{Float64}([]), η, ϕ, ω)
end

function θ2par(θ::Vector{T}, model::INARMAModel)::parameter where {T <: Real}
    θ = Float64.(θ)
    p = length(model.pastObs)
    q = length(model.pastMean)
    r = length(model.external)
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

    nPar = 1 + p + q + r + nb1 + nb2 + zi

    if length(θ) != nPar
        error("Length of θ does not match model specification.")
    end

    β0 = θ[1]
    used = 1
    α = θ[used + 1:used + p]
    used += p
    β = θ[used + 1:used + q]
    used += q
    η = θ[used + 1:used + r]
    used += r
    ϕtemp = θ[used + 1:used + nb1 + nb2]
    if length(ϕtemp) == 2
        ϕ = ϕtemp
    end
    if length(ϕtemp) == 1
        ϕ = repeat(ϕtemp, 2)
    end
    if length(ϕtemp) == 0
        ϕ =  Vector{Float64}([])
    end
    used += nb1 + nb2
    if zi
        ω = θ[end]
    else
        ω = 0.0
    end

    parameter(β0, α, β, η, ϕ, ω)
end

function θ2par(θ::Vector{T}, model::INARModel)::parameter where {T <: Real}
    θ = Float64.(θ)
    p = length(model.pastObs)
    r = length(model.external)
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

    nPar = 1 + p + r + nb1 + nb2 + zi

    if length(θ) != nPar
        error("Length of θ does not match model specification.")
    end

    β0 = θ[1]
    used = 1
    α = θ[used + 1:used + p]
    used += p
    η = θ[used + 1:used + r]
    used += r
    ϕtemp = θ[used + 1:used + nb1 + nb2]
    if length(ϕtemp) == 2
        ϕ = ϕtemp
    end
    if length(ϕtemp) == 1
        ϕ = repeat(ϕtemp, 2)
    end
    if length(ϕtemp) == 0
        ϕ = Vector{Float64}([])
    end
    used += nb1 + nb2
    if zi
        ω = θ[end]
    else
        ω = 0.0
    end

    parameter(β0, α, Vector{Float64}([]), η, ϕ, ω)
end

function θ2par(θ::Vector{T}, model::INMAModel)::parameter where {T <: Real}
    θ = Float64.(θ)
    q = length(model.pastMean)
    r = length(model.external)
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

    nPar = 1 + q + r + nb1 + nb2 + zi

    if length(θ) != nPar
        error("Length of θ does not match model specification.")
    end

    β0 = θ[1]
    used = 1
    β = θ[used + 1:used + q]
    used += q
    η = θ[used + 1:used + r]
    used += r
    ϕtemp = θ[used + 1:used + nb1 + nb2]
    if length(ϕtemp) == 2
        ϕ = ϕtemp
    end
    if length(ϕtemp) == 1
        ϕ = repeat(ϕtemp, 2)
    end
    if length(ϕtemp) == 0
        ϕ = Vector{Float64}([])
    end
    used += nb1 + nb2
    if zi
        ω = θ[end]
    else
        ω = 0.0
    end

    parameter(β0, Vector{Float64}([]), β, η, ϕ, ω)
end

# Idea: Take vector and put restrictions in the correct place
function expandRestrictions(θ::Vector{T1},
    model::T2,
    restr::Vector{Pair{String, T3}}) where {T1, T3<:Real, T2 <: CountModel}
    #
    if length(restr) == 0
        return θ
    end
    symVec = (x -> x[1]).(restr)
    if any(symVec .== "β0")
        i = findfirst(symVec .== "β0")
        out = Float64(restr[i][2])
        if length(restr) == 1
            return [out; θ]
        else
            restr = restr[2:end]
            symVec = (x -> x[1]).(restr)
        end
    else
        out = θ[1]
        θ = θ[2:end]
    end
    
    l = length.(symVec)
    sym = fill(:a, length(l))
    ind = zeros(Int64, length(l))
    vals = (x -> Float64(x[2])).(restr)
    for i = 1:length(l)
        if l[i] > 1
            sym[i] = Symbol(symVec[i][1])
            ind[i] = parse(Int64, last(symVec[i], l[i] - 1))
        else
            sym[i] = Symbol(symVec[i])
        end
    end
    
    if typeof(model) in [INARModel, INARMAModel, INARCHModel, INGARCHModel]
        if any(sym .== :α)
            indα = findall(sym .== :α)
            nr = length(indα)
            temp = fill(-Inf, length(model.pastObs))
            for i = 1:nr
                temp[ind[indα[i]]] = vals[indα[i]]
            end
            for i = 1:length(temp)
                if !isfinite(temp[i])
                    temp[i] = θ[1]
                    θ = θ[2:end]
                end
            end
            out = [out; temp]
        else
            out = [out; θ[1:length(model.pastObs)]]
            θ = θ[length(model.pastObs)+1:end]
        end
    end
    
    if typeof(model) in [INARMAModel, INMAModel, INGARCHModel]
        if any(sym .== :β)
            indβ = findall(sym .== :β)
            nr = length(indβ)
            temp = fill(-Inf, length(model.pastMean))
            for i = 1:nr
                temp[ind[indβ[i]]] = vals[indβ[i]]
            end
            for i = 1:length(temp)
                if !isfinite(temp[i])
                    temp[i] = θ[1]
                    θ = θ[2:end]
                end
            end
        else
            out = [out; θ[1:length(model.pastMean)]]
            θ = θ[length(model.pastMean)+1:end]
        end
    end
    
    
    nReg = size(model.X)[1]
    if any(sym .== :η)
        indη = findall(sym .== :η)
        nr = length(indη)
        temp = fill(-Inf, nReg)
        for i = 1:nr
            temp[ind[indη[i]]] = vals[indη[i]]
        end
        for i = 1:length(temp)
            if !isfinite(temp[i])
                temp[i] = θ[1]
                θ = θ[2:end]
            end
        end
        out = [out; temp]
    else
        out = [out; θ[1:nReg]]
        θ = θ[nReg+1:end]
    end
    
    nPhi = sum(model.distr .== "NegativeBinomial")
    if any(sym .== :ϕ)
        indϕ = findall(sym .== :ϕ)
        nr = length(indϕ)
        temp = fill(-Inf, nPhi)
        for i = 1:nr
            if ind[indϕ[i]] == 0
                temp[1] = vals[indϕ[i]]
            else
                temp[ind[indϕ[i]]] = vals[indϕ[i]]
            end
        end
        for i = 1:length(temp)
            if !isfinite(temp[i])
                temp[i] = θ[1]
                θ = θ[2:end]
            end
        end
        out = [out; temp]
    else
        out = [out; θ[1:nPhi]]
        θ = θ[nPhi+1:end]
    end

    if any(sym .== :ω)
        out = [out; vals[findfirst(sym .== :ω)]]
        return out
    else
        return [out; θ]
    end
end

# Function that takes the full vector of parameters and removes the ones with restrictions
function removeRestrictions(θ::Vector{T1},
    model::T2,
    restr::Vector{Pair{String, T3}}) where {T1, T3<:Real, T2 <: CountModel}
    par = θ2par(θ, model)
    restr2 = Vector{Pair{String,Float64}}(undef, length(restr))
    for i = 1:length(restr)
        restr2[i] = Pair(restr[i][1], Inf)
    end
    out = par2θ(changePar(par, restr2), model)
    out[isfinite.(out)]
end
θ = [10, 0.5, 3]
model = Model(pastObs = 1, distr = "NegativeBinomial")
restr = ["β0" => 9]
removeRestrictions(θ, model, restr)


function θ2par(θ::Vector{T1},
    model::T2,
    restr::Vector{Pair{String,T3}}) where {T1,T3 <: Real, T2 <: CountModel}
    θ2 = expandRestrictions(θ, model, restr)
    return θ2par(θ2, model)
end