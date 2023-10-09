# Function to translate parameter struct vector of parameters

"""
    par2θ(θ, model)
Function two convert parameter struct to vector

* `θ`: Parameter struct
* `model`: Model specification CountModel, INGARCHModel, ...

# Example
```julia-repl
model = CountModel(past_obs = 1) # Specify INARCH(1)
pars = θ2par([10, 0.5], model)
θ = par2θ(pars, model)
```
"""
function par2θ(θ::parameter, model::T)::Vector{Float64} where {T <: CountModel}
    if model <: INARMA
        nϕ = sum(model.distr .== "NegativeBinomial")
        if nϕ == 2
            if length(model.external) == 0
                nϕ -= 1
            else
                if sum(model.external) == 0
                    nϕ -= 1
                end
            end
        end
    end

    if model <: INGARCH
        nϕ = model.distr == "NegativeBinomial"
    end

    [θ.β0; θ.α; θ.β; θ.η; θ.ϕ[1:nϕ]; ifelse(model.zi, θ.ω, Vector{Float64}([]))]
end

function par2θ(θ::parameter, model::T)::Vector{Float64} where {T <: INGARCH}
    nϕ = model.distr == "NegativeBinomial"

    [θ.β0; θ.α; θ.β; θ.η; θ.ϕ[1:nϕ]; ifelse(model.zi, θ.ω, Vector{Float64}([]))]
end

function par2θ(θ::parameter, model::IIDModel)::Vector{Float64}
    nb = model.distr == "NegativeBinomial"
    [θ.β0; θ.η; ifelse(nb, θ.ϕ, Vector{Float64}([])); ifelse(model.zi, θ.ω, Vector{Float64}([]))]
end

function par2θ(θ::parameter, model::INGARCHModel)::Vector{Float64}
    nb = model.distr == "NegativeBinomial"
    [θ.β0; θ.α; θ.β; θ.η; ifelse(nb, θ.ϕ, Vector{Float64}([])); ifelse(model.zi, θ.ω, Vector{Float64}([]))]
end

function par2θ(θ::parameter, model::INARCHModel)::Vector{Float64}
    nb = model.distr == "NegativeBinomial"
    [θ.β0; θ.α; θ.η; ifelse(nb, θ.ϕ, Vector{Float64}([])); ifelse(model.zi, θ.ω, Vector{Float64}([]))]
end

function par2θ(θ::parameter, model::T)::Vector{Float64} where {T <: INARMA}
    nϕ = sum(model.distr .== "NegativeBinomial")
    nϕ = sum(model.distr .== "NegativeBinomial")
    if nϕ == 2
        if length(model.external) == 0
            nϕ -= 1
        else
            if sum(model.external) == 0
                nϕ -= 1
            end
        end
    end
    [θ.β0; θ.α; θ.β; θ.η; θ.ϕ[1:nϕ]; ifelse(model.zi, θ.ω, Vector{Float64}([]))]
end


# With restrictions
function changePar(par::parameter, restr::Vector{Pair{String,T}}) where {T<:Real}
    nR = length(restr)
    out = deepcopy(par)
    pn = propertynames(out)
    for i = 1:nR
        sym = Symbol(restr[i][1])
        if sym in pn
            field = getfield(out, sym)
            if typeof(field) == Vector{Float64}
                setfield!(out, sym, fill(Float64(restr[i][2]), length(field)))
            else
                setfield!(out, sym, Float64(restr[i][2]))
            end
        else
            temp = getfield(out, Symbol(restr[i][1][1]))
            iChange = parse(Int64, restr[i][1][end])
            if iChange > length(temp)
                error("Invalid restriction")
            end
            temp[iChange] = Float64(restr[i][2])
            setfield!(out, Symbol(restr[i][1][1]), temp)
        end
    end
    return out
end

function par2θ(θ::parameter,
               model::T1,
               restr::Vector{Pair{String, T2}})::Vector{Float64} where {T1<:CountModel, T2 <: Real}
    if length(restr) > 0
        θ = changePar(θ, restr)
    end
    return par2θ(θ, model)
end

