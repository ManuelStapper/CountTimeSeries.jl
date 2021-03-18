# Function to check parameters

# Input:
# θ:        parameters
# model:    Model specifications

# Output:
# Admissible parameters or not?!

"""
    parametercheck(θ, model)
Check if parameters are admissable.

* `θ`: Parameters (vector or parameter)
* `model`: Model specification

# Example
```julia-repl
model = Model(pastObs = 1)
parametercheck([10, 0.3], model)
```

Parameters are admissible if they yield strictly positive conditional means and
further fulfill stationarity properties.
"""
function parametercheck(θ::parameter, model::INGARCHModel)
    logl = model.link == "Log"
    lin = !logl
    nb = model.distr == "NegativeBinomial"

    β0 = θ.β0
    α = θ.α
    β = θ.β
    αβ = [α; β]
    η = θ.η
    ϕ = θ.ϕ
    ω = θ.ω

    p = length(α)
    q = length(β)

    if !(0 <= ω <= 1) return false end
    if any(ϕ .<= 0) return false end

    r = length(model.external)

    if r > 0
        if lin
            if !all(η .>= 0)
                return false
            end
        end
    end

    if logl
        if p + q > 0
            if any(abs.(αβ) .>= 1) | (abs(sum(αβ)) >= 1)
                return false
            end
        end
    else
        if β0 <= 0
            return false
        end

        if p + q > 0
            if any(αβ .< 0) | (sum(αβ) > 1)
                return false
            end
        end
    end

    return true
end



function parametercheck(θ::parameter, model::INARCHModel)
    logl = model.link == "Log"
    lin = !logl
    nb = model.distr == "NegativeBinomial"

    β0 = θ.β0
    α = θ.α
    η = θ.η
    ϕ = θ.ϕ
    ω = θ.ω

    p = length(α)

    if !(0 <= ω <= 1) return false end
    if any(ϕ .<= 0) return false end

    r = length(model.external)

    if r > 0
        if lin
            if !all(η .>= 0)
                return false
            end
        end
    end

    if logl
        if p > 0
            if any(abs.(α) .>= 1) | (abs(sum(α)) >= 1)
                return false
            end
        end


    else
        if β0 <= 0
            return false
        end

        if p > 0
            if any(α .< 0) | (sum(α) > 1)
                return false
            end
        end
    end

    return true
end


function parametercheck(θ::parameter, model::IIDModel)
    logl = model.link == "Log"
    lin = !logl
    nb = model.distr == "NegativeBinomial"

    β0 = θ.β0
    η = θ.η
    ϕ = θ.ϕ
    ω = θ.ω

    if !(0 <= ω <= 1) return false end
    if any(ϕ .<= 0) return false end

    r = length(model.external)

    if r > 0
        if lin
            if !all(η .>= 0)
                return false
            end
        end
    end

    if !logl
        if β0 <= 0
            return false
        end
    end

    return true
end




function parametercheck(θ::parameter, model::INARMAModel)
    logl1 = model.link[1] == "Log"
    lin1 = !logl1

    logl2 = model.link[2] == "Log"
    lin2 = !logl2

    nb1 = model.distr[1] == "NegativeBinomial"
    nb2 = model.distr[2] == "NegativeBinomial"

    p = length(model.pastObs)
    q = length(model.pastMean)
    r = length(model.external)

    β0 = θ.β0
    α = θ.α
    β = θ.β
    η = θ.η
    ϕ = θ.ϕ
    ω = θ.ω

    if lin1 & (β0 <= 0)
        return false
    end

    if p > 0
        if any(α .< 0)
            return false
        end

        if sum(α) >= 1
            return false
        end
    end

    if q > 0
        if !all(0 .<= β .< 1)
            return false
        end
    end

    if r > 0
        if lin1
            if !all(η[findall(.!model.external)] .>= 0)
                return false
            end
        end

        if lin2
            if !all(η[findall(model.external)] .>= 0)
                return false
            end
        end
    end

    if nb1 | nb2
        if any(ϕ .<= 0)
            return false
        end
    end

    if !(0 <= ω <= 1)
        return false
    end

    return true
end

function parametercheck(θ::parameter, model::INARModel)
    logl1 = model.link[1] == "Log"
    lin1 = !logl1

    logl2 = model.link[2] == "Log"
    lin2 = !logl2

    nb1 = model.distr[1] == "NegativeBinomial"
    nb2 = model.distr[2] == "NegativeBinomial"

    p = length(model.pastObs)
    r = length(model.external)

    β0 = θ.β0
    α = θ.α
    η = θ.η
    ϕ = θ.ϕ
    ω = θ.ω

    if lin1 & (β0 <= 0)
        return false
    end

    if p > 0
        if any(α .< 0)
            return false
        end

        if sum(α) >= 1
            return false
        end
    end

    if r > 0
        if lin1
            if !all(η[findall(.!model.external)] .>= 0)
                return false
            end
        end

        if lin2
            if !all(η[findall(model.external)] .>= 0)
                return false
            end
        end
    end

    if nb1 | nb2
        if any(ϕ .<= 0)
            return false
        end
    end

    if !(0 <= ω <= 1)
        return false
    end

    return true
end

function parametercheck(θ::parameter, model::INMAModel)
    logl1 = model.link[1] == "Log"
    lin1 = !logl1

    logl2 = model.link[2] == "Log"
    lin2 = !logl2

    nb1 = model.distr[1] == "NegativeBinomial"
    nb2 = model.distr[2] == "NegativeBinomial"

    q = length(model.pastMean)
    r = length(model.external)

    β0 = θ.β0
    β = θ.β
    η = θ.η
    ϕ = θ.ϕ
    ω = θ.ω

    if lin1 & (β0 <= 0)
        return false
    end

    if q > 0
        if !all(0 .<= β .< 1)
            return false
        end
    end

    if r > 0
        if lin1
            if !all(η[findall(.!model.external)] .>= 0)
                return false
            end
        end

        if lin2
            if !all(η[findall(model.external)] .>= 0)
                return false
            end
        end
    end

    if nb1 | nb2
        if any(ϕ .<= 0)
            return false
        end
    end

    if !(0 <= ω <= 1)
        return false
    end

    return true
end

function parametercheck(θ::Array{T, 1} where T<: AbstractFloat, model::T where T<:CountModel)
    parametercheck(θ2par(θ, model), model)
end
