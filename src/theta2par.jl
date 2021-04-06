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
function θ2par(θ::Array{T, 1} where T<: AbstractFloat, model::INGARCHModel)
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
        ϕ = Array{Float64, 1}([])
    end
    if zi
        ω = θ[end]
    else
        ω = 0.0
    end

    parameter(β0, α, β, η, ϕ, ω)
end


function θ2par(θ::Array{T, 1} where T<: AbstractFloat, model::INARCHModel)
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
        ϕ = Array{Float64, 1}([])
    end
    if zi
        ω = θ[end]
    else
        ω = 0.0
    end

    parameter(β0, α, Array{Float64, 1}([]), η, ϕ, ω)
end

function θ2par(θ::Array{T, 1} where T<: AbstractFloat, model::IIDModel)
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
        ϕ = Array{Float64, 1}([])
    end
    if zi
        ω = θ[end]
    else
        ω = 0.0
    end

    parameter(β0, Array{Float64, 1}([]), Array{Float64, 1}([]), η, ϕ, ω)
end

function θ2par(θ::Array{T, 1} where T<: AbstractFloat, model::INARMAModel)
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
        ϕ =  Array{Float64, 1}([])
    end
    used += nb1 + nb2
    if zi
        ω = θ[end]
    else
        ω = 0.0
    end

    parameter(β0, α, β, η, ϕ, ω)
end

function θ2par(θ::Array{T, 1} where T<: AbstractFloat, model::INARModel)
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
        ϕ = Array{Float64, 1}([])
    end
    used += nb1 + nb2
    if zi
        ω = θ[end]
    else
        ω = 0.0
    end

    parameter(β0, α, Array{Float64, 1}([]), η, ϕ, ω)
end

function θ2par(θ::Array{T, 1} where T<: AbstractFloat, model::INMAModel)
    q = length(model.pastMean)
    r = length(model.external)
    nb1 = model.distr[1] == "NegativeBinomial"
    nb2 = model.distr[2] == "NegativeBinomial"
    if r == 0
        nb2 = false
    else
        if sum(.!model.external) > 0
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
        ϕ = Array{Float64, 1}([])
    end
    used += nb1 + nb2
    if zi
        ω = θ[end]
    else
        ω = 0.0
    end

    parameter(β0, Array{Float64, 1}([]), β, η, ϕ, ω)
end

function θ2par(θ::Array{T, 1} where T<: Integer, model::T where T<:CountModel)
    θ2par(convert(Vector{Float64}, θ), model)
end
