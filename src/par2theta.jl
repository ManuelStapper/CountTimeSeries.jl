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
function par2θ(θ::parameter, model::T where T<:CountModel)
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

    [θ.β0; θ.α; θ.β; θ.η; θ.ϕ[1:nϕ]; ifelse(model.zi, θ.ω, Array{Float64, 1}([]))]
end

function par2θ(θ::parameter, model::T where T<:INGARCH)
    nϕ = model.distr == "NegativeBinomial"

    [θ.β0; θ.α; θ.β; θ.η; θ.ϕ[1:nϕ]; ifelse(model.zi, θ.ω, Array{Float64, 1}([]))]
end

function par2θ(θ::parameter, model::IIDModel)
    nb = model.distr == "NegativeBinomial"
    [θ.β0; θ.η; ifelse(nb, θ.ϕ, Array{Float64, 1}([])); ifelse(model.zi, θ.ω, Array{Float64, 1}([]))]
end

function par2θ(θ::parameter, model::INGARCHModel)
    nb = model.distr == "NegativeBinomial"
    [θ.β0; θ.α; θ.β; θ.η; ifelse(nb, θ.ϕ, Array{Float64, 1}([])); ifelse(model.zi, θ.ω, Array{Float64, 1}([]))]
end

function par2θ(θ::parameter, model::INARCHModel)
    nb = model.distr == "NegativeBinomial"
    [θ.β0; θ.α; θ.η; ifelse(nb, θ.ϕ, Array{Float64, 1}([])); ifelse(model.zi, θ.ω, Array{Float64, 1}([]))]
end

function par2θ(θ::parameter, model::T where T<:INARMA)
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
    [θ.β0; θ.α; θ.β; θ.η; θ.ϕ[1:nϕ]; ifelse(model.zi, θ.ω, Array{Float64, 1}([]))]
end
