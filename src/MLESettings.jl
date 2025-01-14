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
function MLESettings(y::Vector{Int64},
                     model::INGARCHModel,
                     init::Vector{T1} = Vector{Float64}([]);
                     optimizer::String = "BFGS",
                     ci::Bool = false,
                     maxEval::T2 = 1e10)::MLEControl where {T1, T2 <: Real}
    p = length(model.pastObs)
    q = length(model.pastMean)
    r = length(model.external)
    nb = model.distr == "NegativeBinomial"
    gp = model.distr == "GPoisson"
    maxEval = Int64(round(maxEval))

    nPar = 1 + p + q + r + nb + gp + model.zi
    init = Vector{Float64}(init)

    if (length(init) != nPar) & (length(init) > 0)
        println("Number of initial values does not match number of parameters.")
        println("Switched to default initial values.")
        init = Vector{Float64}(undef, 0)
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
            temp = Vector{Float64}([])
        end

        init = [β0init; temp; fill(0.05, r); fill(3.0, nb + gp); ifelse(model.zi, 0.1, Vector{Float64}([]))]
    end

    init = θ2par(init, model)

    if !(optimizer in ["BFGS", "LBFGS", "NelderMead"])
        println("Invalid optimizing method. Set to BFGS.")
        optimizer = "BFGS"
    end

    MLEControl(init, optimizer, ci, Int64(round(abs(maxEval))))
end

function MLESettings(y::Vector{Int64},
                     model::INARCHModel,
                     init::Vector{T1} = Vector{Float64}([]);
                     optimizer::String = "BFGS",
                     ci::Bool = false,
                     maxEval::T2 = 1e10)::MLEControl where {T1, T2 <: Real}

    p = length(model.pastObs)
    r = length(model.external)
    nb = model.distr == "NegativeBinomial"
    gp = model.distr == "GPoisson"

    nPar = 1 + p + r + nb + gp + model.zi
    init = Vector{Float64}(init)
    maxEval = Int64(round(maxEval))

    if (length(init) != nPar) & (length(init) > 0)
        println("Number of initial values does not match number of parameters.")
        println("Switched to default initial values.")
        init = Vector{Float64}(undef, 0)
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
            temp = Vector{Float64}([])
        end

        init = [β0init; temp; fill(0.05, r); fill(3.0, nb + gp); ifelse(model.zi, 0.1, Vector{Float64}([]))]
    end

    init = θ2par(init, model)

    if !(optimizer in ["BFGS", "LBFGS", "NelderMead"])
        println("Invalid optimizing method. Set to BFGS.")
        optimizer = "BFGS"
    end

    MLEControl(init, optimizer, ci, Int64(round(abs(maxEval))))
end

function MLESettings(y::Vector{Int64},
                     model::IIDModel,
                     init::Vector{T1} = Vector{Float64}([]);
                     optimizer::String = "BFGS",
                     ci::Bool = false,
                     maxEval::T2 = 1e10)::MLEControl where {T1, T2 <: Real}
    r = length(model.external)
    nb = model.distr == "NegativeBinomial"
    gp = model.distr == "GPoisson"

    nPar = 1 + r + nb + gp + model.zi
    init = Vector{Float64}(init)
    maxEval = Int64(round(maxEval))

    if (length(init) != nPar) & (length(init) > 0)
        println("Number of initial values does not match number of parameters.")
        println("Switched to default initial values.")
        init = Vector{Float64}(undef, 0)
    end

    if length(init) == 0
        β0init = mean(y)/2
        if model.link == "Log"
            β0init = log(β0init)
        end

        init = [β0init; fill(0.05, r); fill(3.0, nb + gp); ifelse(model.zi, 0.1, Vector{Float64}([]))]
    end

    init = θ2par(init, model)

    if !(optimizer in ["BFGS", "LBFGS", "NelderMead"])
        println("Invalid optimizing method. Set to BFGS.")
        optimizer = "BFGS"
    end

    MLEControl(init, optimizer, ci, Int64(round(abs(maxEval))))
end

function MLESettings(y::Vector{Int64},
                     model::INARMAModel,
                     init::Vector{T1} = Vector{Float64}([]);
                     optimizer::String = "BFGS",
                     ci::Bool = false,
                     maxEval::T2 = 1e10)::MLEControl where {T1, T2 <: Real}

    p = length(model.pastObs)
    q = length(model.pastMean)
    r = length(model.external)
    maxEval = Int64(round(maxEval))
    nϕ = sum(model.distr .== "NegativeBinomial") + sum(model.distr .== "GPoisson")
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
    init = Vector{Float64}(init)

    if (length(init) != nPar) & (length(init) > 0)
        println("Number of initial values does not match number of parameters.")
        println("Switched to default initial values.")
        init = Vector{Float64}(undef, 0)
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
            temp = Vector{Float64}([])
        end

        init = [β0init; temp; fill(0.05, r); fill(3.0, nϕ); ifelse(model.zi, 0.1, Vector{Float64}([]))]
    end

    init = θ2par(init, model)

    if !(optimizer in ["BFGS", "LBFGS", "NelderMead"])
        println("Invalid optimizing method. Set to BFGS.")
        optimizer = "BFGS"
    end

    MLEControl(init, optimizer, ci, Int64(round(abs(maxEval))))
end

function MLESettings(y::Vector{Int64},
                     model::INARModel,
                     init::Vector{T1} = Vector{Float64}([]);
                     optimizer::String = "BFGS",
                     ci::Bool = false,
                     maxEval::T2 = 1e10)::MLEControl where {T1, T2 <: Real}

    p = length(model.pastObs)
    r = length(model.external)
    maxEval = Int64(round(maxEval))
    nϕ = sum(model.distr .== "NegativeBinomial") + sum(model.distr .== "GPoisson")
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
    init = Vector{Float64}(init)

    if (length(init) != nPar) & (length(init) > 0)
        println("Number of initial values does not match number of parameters.")
        println("Switched to default initial values.")
        init = Vector{Float64}(undef, 0)
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
            temp = Vector{Float64}([])
        end

        init = [β0init; temp; fill(0.05, r); fill(3.0, nϕ); ifelse(model.zi, 0.1, Vector{Float64}([]))]
    end

    init = θ2par(init, model)

    if !(optimizer in ["BFGS", "LBFGS", "NelderMead"])
        println("Invalid optimizing method. Set to BFGS.")
        optimizer = "BFGS"
    end

    MLEControl(init, optimizer, ci, Int64(round(abs(maxEval))))
end

function MLESettings(y::Vector{Int64},
                     model::INMAModel,
                     init::Vector{T1} = Vector{Float64}([]);
                     optimizer::String = "BFGS",
                     ci::Bool = false,
                     maxEval::T2 = 1e10)::MLEControl where {T1, T2 <: Real}

    q = length(model.pastMean)
    r = length(model.external)
    maxEval = Int64(round(maxEval))
    nϕ = sum(model.distr .== "NegativeBinomial") + sum(model.distr .== "GPoisson")
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
    init = Vector{Float64}(init)

    if (length(init) != nPar) & (length(init) > 0)
        println("Number of initial values does not match number of parameters.")
        println("Switched to default initial values.")
        init = Vector{Float64}(undef, 0)
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
            temp = Vector{Float64}([])
        end

        init = [β0init; temp; fill(0.05, r); fill(3.0, nϕ); ifelse(model.zi, 0.1, Vector{Float64}([]))]
    end

    init = θ2par(init, model)

    if !(optimizer in ["BFGS", "LBFGS", "NelderMead"])
        println("Invalid optimizing method. Set to BFGS.")
        optimizer = "BFGS"
    end

    MLEControl(init, optimizer, ci, Int64(round(abs(maxEval))))
end

function MLESettings(y::Vector{Int64},
                     model::T1,
                     init::parameter;
                     optimizer::String = "BFGS",
                     ci::Bool = false,
                     maxEval::T2 = 1e10)::MLEControl where {T1 <: CountModel, T2 <: Real}
    initNew = par2θ(init, model)
    maxEval = Int64(round(maxEval))
    MLESettings(y, model, initNew, optimizer = optimizer, ci = ci, maxEval = maxEval)
end
