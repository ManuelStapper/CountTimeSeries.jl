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
        nϕ = sum(model.distr .== "NegativeBinomial") + sum(model.distr .== "GPoisson")
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
        nϕ = (model.distr == "NegativeBinomial") + (model.distr == "GPoisson")
    end

    [θ.β0; θ.α; θ.β; θ.η; θ.ϕ[1:nϕ]; ifelse(model.zi, θ.ω, Vector{Float64}([]))]
end

function par2θ(θ::parameter, model::T)::Vector{Float64} where {T <: INGARCH}
    nϕ = (model.distr == "NegativeBinomial") + (model.distr == "GPoisson")

    [θ.β0; θ.α; θ.β; θ.η; θ.ϕ[1:nϕ]; ifelse(model.zi, θ.ω, Vector{Float64}([]))]
end

function par2θ(θ::parameter, model::IIDModel)::Vector{Float64}
    nϕ = (model.distr == "NegativeBinomial") + (model.distr == "GPoisson")
    [θ.β0; θ.η; ifelse(nϕ > 0, θ.ϕ, Vector{Float64}([])); ifelse(model.zi, θ.ω, Vector{Float64}([]))]
end

function par2θ(θ::parameter, model::INGARCHModel)::Vector{Float64}
    nϕ = (model.distr == "NegativeBinomial") + (model.distr == "GPoisson")
    [θ.β0; θ.α; θ.β; θ.η; ifelse(nϕ > 0, θ.ϕ, Vector{Float64}([])); ifelse(model.zi, θ.ω, Vector{Float64}([]))]
end

function par2θ(θ::parameter, model::INARCHModel)::Vector{Float64}
    nϕ = (model.distr == "NegativeBinomial") + (model.distr == "GPoisson")
    [θ.β0; θ.α; θ.η; ifelse(nϕ > 0, θ.ϕ, Vector{Float64}([])); ifelse(model.zi, θ.ω, Vector{Float64}([]))]
end

function par2θ(θ::parameter, model::T)::Vector{Float64} where {T <: INARMA}
    nϕ = sum(model.distr .== "NegativeBinomial") + sum(model.distr .== "GPoisson")
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

# Create function that takes a parameter object and restrictions and replaces the restricted elements
function changePar(par::parameter, restr::Vector{Pair{String,T1}}, model::T2) where {T1<:Real, T2 <: CountModel}
    # Number of restrictions
    nR = length(restr)
    # Create copy to update for output
    out = deepcopy(par)
    # Vector of parameter names
    pn = propertynames(out)
    # Iterate over all restrictions
    for i = 1:nR
        # Which parameter is affected by the restriction?
        # One of: :β0, :α, :β, :η, :ϕ, :ω
        sym = Symbol(restr[i][1])
        if sym in pn # If the parameter name is not indexed (For ϕ and ω)
            field = getfield(out, sym)
            # If the replacement type is a vector (for example for ϕ)
            if typeof(field) == Vector{Float64}
                setfield!(out, sym, fill(Float64(restr[i][2]), length(field)))
            else
                # Otherwise, replace just one value
                setfield!(out, sym, Float64(restr[i][2]))
            end
        else # If the parameter name contains index (such as α1 or β10 or such)
            # Get the corresponding complete vector
            temp = getfield(out, Symbol(restr[i][1][1]))

            # Find out which value to replace

            # iOrder is the value given in the restriction: "α7" -> 7, might be 2 digit number
            iOrder = parse(Int64, last(restr[i][1], length(restr[i][1]) - 1))

            if string(sym)[1] == 'α'
                iChange = findfirst(model.pastObs .== iOrder)
            elseif string(sym)[1] == 'β'
                iChange = findfirst(model.pastMean .== iOrder)
            else
                iChange = iOrder
            end
            if iChange > length(temp)
                error("Invalid restriction")
            end
            # Replace value and update in parameter object
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
        θ = changePar(θ, restr, model)
    end
    return par2θ(θ, model)
end

