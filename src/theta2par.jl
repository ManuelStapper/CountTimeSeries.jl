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
