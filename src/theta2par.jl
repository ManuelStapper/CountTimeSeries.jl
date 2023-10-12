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

# Idea: Take vector of free parameters and put restrictions in the correct place
#       to get the "full" parameter vector
function expandRestrictions(θ::Vector{T1},
    model::T2,
    restr::Vector{Pair{String, T3}}) where {T1, T3<:Real, T2 <: CountModel}
    # If no restriction, nothing happens
    if length(restr) == 0
        return θ
    end
    # Find restricted parameters
    symVec = (x -> x[1]).(restr)

    # Special case intercept parameter β0 (because of notation)
    if any(symVec .== "β0")
        # If it is restricted, find the corresponding restriction
        i = findfirst(symVec .== "β0")
        # Start the output vector of all parameters
        out = Float64(restr[i][2])
        if length(restr) == 1
            # If that was the only restriction, done
            return [out; θ]
        else
            # Otherwise, remove the intercept parameter restriction
            restr = restr[setdiff(1:length(restr), i)]
            # Update the names of restricted parameters
            symVec = (x -> x[1]).(restr)
        end
    else
        # If β0 is not restricted, it is the first element of θ
        out = θ[1]
        θ = θ[2:end]
    end
    
    # Compute the length of paramter names, to find out if a certain parameter is restricted (such as α1)
    # or a parameter without "order" (like ω)
    l = length.(symVec)
    # Then split the parameter name into the parameter class "sym", the index "ind"
    # and the restriction value "vals"
    # "α2" => 0.5   -->   :α, 2 and 0.5 
    sym = fill(:a, length(l))
    ind = zeros(Int64, length(l))
    vals = (x -> Float64(x[2])).(restr)
    for i = 1:length(l)
        if l[i] > 1
            # If it has ordering number
            sym[i] = Symbol(symVec[i][1])
            ind[i] = parse(Int64, last(symVec[i], l[i] - 1))
        else
            sym[i] = Symbol(symVec[i])
        end
    end
    
    # If there are restrictions of α parameters, check first if the model contains those
    if typeof(model) in [INARModel, INARMAModel, INARCHModel, INGARCHModel]
        if any(sym .== :α)
            # If one or more α parameters are restricted, find out which restrictions
            indα = findall(sym .== :α)
            # Number of α restrictions
            nr = length(indα)
            # Initialise the α parameter vector
            temp = fill(-Inf, length(model.pastObs))
            # Fill in all values in the α vector that come from restrictions
            for i = 1:nr
                temp[ind[indα[i]]] = vals[indα[i]]
            end
            # Then fill in the values from "free" parameters
            for i = 1:length(temp)
                if !isfinite(temp[i])
                    temp[i] = θ[1]
                    θ = θ[2:end]
                end
            end
            out = [out; temp]
        else
            # If there are no restrictions, just add the free parameters to the output
            # and remove them from input
            out = [out; θ[1:length(model.pastObs)]]
            θ = θ[length(model.pastObs)+1:end]
        end
    end
    
    # Same procedure for β, η and ϕ parameters as for α parameters
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

function θ2par(θ::Vector{T1},
    model::T2,
    restr::Vector{Pair{String,T3}}) where {T1,T3 <: Real, T2 <: CountModel}
    θ2 = expandRestrictions(θ, model, restr)
    return θ2par(θ2, model)
end