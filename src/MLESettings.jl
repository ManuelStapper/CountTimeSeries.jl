# Constructor for MLEControl struct

"""
    MLESettings(y, model, init, optimizer, ci)
Wrapper function to specify estimation settings.

* `y`: Time series
* `model`: Model specification
* `init`: Initial values (vector or parameter).
* `optimizer`: Optimization routine. "BFGS", "LBFGS" or "NelderMead"
* `ci`: Indicator: Shall confidence intervals be computed?
* `maxEval`: Maximum number of likelihood evaluations

# Example
```julia-repl
MLESettings(y, model, ci = true, maxEval = 1e9)
```

If the argument `init` is not given, valid initial values are chosen.
See also [MLEControl](@ref).
"""
function MLESettings(y::Array{T, 1} where T<:Integer,
    model::INGARCHModel,
    init::Array{T, 1} where T<:AbstractFloat = Array{Float64, 1}([]);
    optimizer::String = "BFGS",
    ci::Bool = false,
    maxEval::T where T<:AbstractFloat = 1e10)

    p = length(model.pastObs)
    q = length(model.pastMean)
    r = length(model.external)
    nb = model.distr == "NegativeBinomial"

    nPar = 1 + p + q + r + nb + model.zi

    if (length(init) != nPar) & (length(init) > 0)
        println("Number of initial values does not match number of parameters.")
        println("Switched to default initial values.")
        init = Array{Float64, 1}(undef, 0)
    end

    if length(init) == 0
        β0init = mean(y)/2
        if model.link == "Log"
            β0init = log(β0init)
        end

        M = p + q
        if M > 0
            temp = [collect(p:-1:1); collect(q:-1:1)]
            temp = temp./(2*sum(temp))
        else
            temp = Array{Float64, 1}([])
        end

        init = [β0init; temp; fill(0.05, r); fill(5.0, nb); ifelse(model.zi, 0.1, Array{Float64, 1}([]))]
    end

    init = θ2par(init, model)

    if !(optimizer in ["BFGS", "LBFGS", "NelderMead"])
        println("Invalid optimizing method. Set to BFGS.")
        optimizer = "BFGS"
    end

    MLEControl(init, optimizer, ci, Int64(round(abs(maxEval))))
end

function MLESettings(y::Array{T, 1} where T<:Integer,
    model::INARCHModel,
    init::Array{T, 1} where T<:AbstractFloat = Array{Float64, 1}([]);
    optimizer::String = "BFGS",
    ci::Bool = false,
    maxEval::T where T<:AbstractFloat = 1e10)

    p = length(model.pastObs)
    r = length(model.external)
    nb = model.distr == "NegativeBinomial"

    nPar = 1 + p + r + nb + model.zi

    if (length(init) != nPar) & (length(init) > 0)
        println("Number of initial values does not match number of parameters.")
        println("Switched to default initial values.")
        init = Array{Float64, 1}(undef, 0)
    end

    if length(init) == 0
        β0init = mean(y)/2
        if model.link == "Log"
            β0init = log(β0init)
        end

        if p > 0
            temp = collect(p:-1:1)
            temp = temp./(2*sum(temp))
        else
            temp = Array{Float64, 1}([])
        end

        init = [β0init; temp; fill(0.05, r); fill(5.0, nb); ifelse(model.zi, 0.1, Array{Float64, 1}([]))]
    end

    init = θ2par(init, model)

    if !(optimizer in ["BFGS", "LBFGS", "NelderMead"])
        println("Invalid optimizing method. Set to BFGS.")
        optimizer = "BFGS"
    end

    MLEControl(init, optimizer, ci, Int64(round(abs(maxEval))))
end

function MLESettings(y::Array{T, 1} where T<:Integer,
    model::IIDModel,
    init::Array{T, 1} where T<:AbstractFloat = Array{Float64, 1}([]);
    optimizer::String = "BFGS",
    ci::Bool = false,
    maxEval::T where T<:AbstractFloat = 1e10)

    r = length(model.external)
    nb = model.distr == "NegativeBinomial"

    nPar = 1 + r + nb + model.zi

    if (length(init) != nPar) & (length(init) > 0)
        println("Number of initial values does not match number of parameters.")
        println("Switched to default initial values.")
        init = Array{Float64, 1}(undef, 0)
    end

    if length(init) == 0
        β0init = mean(y)/2
        if model.link == "Log"
            β0init = log(β0init)
        end

        init = [β0init; fill(0.05, r); fill(5.0, nb); ifelse(model.zi, 0.1, Array{Float64, 1}([]))]
    end

    init = θ2par(init, model)

    if !(optimizer in ["BFGS", "LBFGS", "NelderMead"])
        println("Invalid optimizing method. Set to BFGS.")
        optimizer = "BFGS"
    end

    MLEControl(init, optimizer, ci, Int64(round(abs(maxEval))))
end

function MLESettings(y::Array{T, 1} where T<:Integer,
    model::INARMAModel,
    init::Array{T, 1} where T<:AbstractFloat = Array{Float64, 1}([]);
    optimizer::String = "BFGS",
    ci::Bool = false,
    maxEval::T where T<:AbstractFloat = 1e10)

    p = length(model.pastObs)
    q = length(model.pastMean)
    r = length(model.external)
    nϕ = sum(model.distr .== "NegativeBinomial")
    if nϕ == 2
        if r == 0
            nϕ -= 1
        else
            if sum(model.external) == 0
                nϕ -= 1
            end
        end
    end

    nPar = 1 + p + q + r + nϕ + model.zi

    if (length(init) != nPar) & (length(init) > 0)
        println("Number of initial values does not match number of parameters.")
        println("Switched to default initial values.")
        init = Array{Float64, 1}(undef, 0)
    end

    if length(init) == 0
        β0init = mean(y)/2
        if model.link[1] == "Log"
            β0init = log(β0init)
        end

        M = p + q
        if M > 0
            temp = [collect(p:-1:1); collect(q:-1:1)]
            temp = temp./(2*sum(temp))
        else
            temp = Array{Float64, 1}([])
        end

        init = [β0init; temp; fill(0.05, r); fill(5.0, nϕ); ifelse(model.zi, 0.1, Array{Float64, 1}([]))]
    end

    init = θ2par(init, model)

    if !(optimizer in ["BFGS", "LBFGS", "NelderMead"])
        println("Invalid optimizing method. Set to BFGS.")
        optimizer = "BFGS"
    end

    MLEControl(init, optimizer, ci, Int64(round(abs(maxEval))))
end

function MLESettings(y::Array{T, 1} where T<:Integer,
    model::INARModel,
    init::Array{T, 1} where T<:AbstractFloat = Array{Float64, 1}([]);
    optimizer::String = "BFGS",
    ci::Bool = false,
    maxEval::T where T<:AbstractFloat = 1e10)

    p = length(model.pastObs)
    r = length(model.external)
    nϕ = sum(model.distr .== "NegativeBinomial")
    if nϕ == 2
        if r == 0
            nϕ -= 1
        else
            if sum(model.external) == 0
                nϕ -= 1
            end
        end
    end

    nPar = 1 + p + r + nϕ + model.zi

    if (length(init) != nPar) & (length(init) > 0)
        println("Number of initial values does not match number of parameters.")
        println("Switched to default initial values.")
        init = Array{Float64, 1}(undef, 0)
    end

    if length(init) == 0
        β0init = mean(y)/2
        if model.link[1] == "Log"
            β0init = log(β0init)
        end

        if p > 0
            temp = collect(p:-1:1)
            temp = temp./(2*sum(temp))
        else
            temp = Array{Float64, 1}([])
        end

        init = [β0init; temp; fill(0.05, r); fill(5.0, nϕ); ifelse(model.zi, 0.1, Array{Float64, 1}([]))]
    end

    init = θ2par(init, model)

    if !(optimizer in ["BFGS", "LBFGS", "NelderMead"])
        println("Invalid optimizing method. Set to BFGS.")
        optimizer = "BFGS"
    end

    MLEControl(init, optimizer, ci, Int64(round(abs(maxEval))))
end

function MLESettings(y::Array{T, 1} where T<:Integer,
    model::INMAModel,
    init::Array{T, 1} where T<:AbstractFloat = Array{Float64, 1}([]);
    optimizer::String = "BFGS",
    ci::Bool = false,
    maxEval::T where T<:AbstractFloat = 1e10)

    q = length(model.pastMean)
    r = length(model.external)
    nϕ = sum(model.distr .== "NegativeBinomial")
    if nϕ == 2
        if r == 0
            nϕ -= 1
        else
            if sum(model.external) == 0
                nϕ -= 1
            end
        end
    end


    nPar = 1 + q + r + nϕ + model.zi

    if (length(init) != nPar) & (length(init) > 0)
        println("Number of initial values does not match number of parameters.")
        println("Switched to default initial values.")
        init = Array{Float64, 1}(undef, 0)
    end

    if length(init) == 0
        β0init = mean(y)/2
        if model.link[1] == "Log"
            β0init = log(β0init)
        end

        if q > 0
            temp = collect(q:-1:1)
            temp = temp./(2*sum(temp))
        else
            temp = Array{Float64, 1}([])
        end

        init = [β0init; temp; fill(0.05, r); fill(5.0, nϕ); ifelse(model.zi, 0.1, Array{Float64, 1}([]))]
    end

    init = θ2par(init, model)

    if !(optimizer in ["BFGS", "LBFGS", "NelderMead"])
        println("Invalid optimizing method. Set to BFGS.")
        optimizer = "BFGS"
    end

    MLEControl(init, optimizer, ci, Int64(round(abs(maxEval))))
end

function MLESettings(y::Array{T, 1} where T<:Integer,
    model::T where T<:CountModel,
    init::parameter;
    optimizer::String = "BFGS",
    ci::Bool = false,
    maxEval::T where T<:AbstractFloat = 1e10)

    initNew = par2θ(init, model)
    MLESettings(y, model, initNew, optimizer = optimizer, ci = ci, maxEval = maxEval)
end
