# Function to create a Count Time Series Model specification object
# Input similar to Mspec, but allows for different inputs

"""
    Model(;model = "INGARCH", distr = "Poisson", link = "Linear",
          pastObs = [], pastMean = [], X = [],
          external = [], zi = false)
Wrapper function to define count data models.

Input defined as in [ModelTypes](@ref). Default setting is an an IID Poisson process without regressors or zero inflation.

# Examples
```julia-repl
Model(pastObs = 1:2, pastMean = 1) # INGARCH(2, 1)
Model(pastObs = [1, 2], distr = "NegativeBinomial") # NB-INARCH(2)
Model(model = "INARMA", pastMean = 1, zi = true) # Zero inflated INMA(1)
```

For details, see [Documentation](https://github.com/ManuelStapper/CountTimeSeries.jl/blob/master/CountTimeSeries_documentation.pdf)
"""
function Model(;model = "INGARCH",
    distr = "Poisson",
    link = "Linear",
    pastObs = Array{Int64, 1}([]),
    pastMean = Array{Int64, 1}([]),
    X = Array{Float64, 2}(undef, 0, 0),
    external = Array{Bool, 1}([]),
    zi = false)

    if !(model in ["INARMA", "INGARCH"])
        error("Invalid model type.")
    end

    if typeof(distr) == String
        if !(distr in ["Poisson", "NegativeBinomial"])
            error("Invalid distribution.")
        else
            distr = fill(distr, 2)
        end
    elseif typeof(distr) == Array{String, 1}
        if length(distr) == 0
            error("Invalid distribution.")
        end
        if length(distr) == 1
            distr = fill(distr[1], 2)
        end
        if length(distr) > 2
            distr = distr[1:2]
        end

        if !(distr[1] in ["Poisson", "NegativeBinomial"])
            error("Invalid (first) distribution.")
        end

        if !(distr[2] in ["Poisson", "NegativeBinomial"])
            error("Invalid (second) distribution.")
        end
    else
        error("Invalid distribution")
    end

    if typeof(link) == String
        if !(link in ["Linear", "Log"])
            error("Invalid link.")
        else
            link = fill(link, 2)
        end
    elseif typeof(link) == Array{String, 1}
        if length(link) == 0
            error("Invalid link.")
        end
        if length(link) == 1
            link = fill(link[1], 2)
        end
        if length(link) > 2
            link = link[1:2]
        end

        if !(link[1] in ["Linear", "Log"])
            error("Invalid (first) link.")
        end

        if !(link[2] in ["Linear", "Log"])
            error("Invalid (second) link.")
        end
    else
        error("Invalid link")
    end

    if typeof(pastObs) == UnitRange{Int64}
        pastObs = collect(pastObs)
    end

    if length(pastObs) == 0
        pastObs = Array{Int64, 1}([])
    end

    if typeof(pastObs) <: Integer
        pastObs = [pastObs]
    end

    if typeof(pastObs) <: Matrix{T} where T<:Integer
        if size(pastObs)[1] == 1
            pastObs = pastObs[1, :]
        elseif size(pastObs)[2] == 1
            pastObs = pastObs[:, 1]
        else
            error("Invalid specification of pastObs.")
        end
    end

    if !(typeof(pastObs) <: Array{T, 1} where T<:Integer)
        error("Invalid specification of pastObs.")
    else
        if any(pastObs .<= 0)
            error("Entries of pastObs must be positive integers.")
        else
            pastObs = sort(unique(pastObs))
        end
    end

    if typeof(pastMean) == UnitRange{Int64}
        pastMean = collect(pastMean)
    end

    if length(pastMean) == 0
        pastMean = Array{Int64, 1}([])
    end

    if typeof(pastMean) <: Integer
        pastMean = [pastMean]
    end

    if typeof(pastMean) <: Matrix{T} where T<:Integer
        if size(pastMean)[1] == 1
            pastMean = pastMean[1, :]
        elseif size(pastMean)[2] == 1
            pastMean = pastMean[:, 1]
        else
            error("Invalid specification of pastMean.")
        end
    end

    if typeof(pastMean) != Array{Int64, 1}
        error("Invalid specification of pastMean.")
    else
        if any(pastMean .<= 0)
            error("Entries of pastMean must be positive integers.")
        else
            pastMean = sort(unique(pastMean))
        end
    end

    if length(external) == 0
        external = Array{Bool, 1}([])
    end

    if typeof(external) == Bool
        external = [external]
    end

    if typeof(external) == Array{Bool, 2}
        if size(external)[1] == 1
            external = external[1, :]
        elseif size(external)[2] == 1
            external = external[:, 1]
        else
            error("Invalid specification of external.")
        end
    end

    if typeof(external) != Array{Bool, 1}
        error("Invalid specification of external.")
    end

    r = length(external)

    X = X .+ 0.0

    if typeof(X) == UnitRange{Int64}
        X = reshape(collect(X), (1, length(X)))
    end

    if length(X) == 0
        X = zeros(0, 0)
        if r > 0
            error("Invalid specification of external and X.")
        end
    else
        if typeof(X) <: Vector{T} where T<:AbstractFloat
            if r > 1
                error("Invalid specification of external and X.")
            end
            if r == 0
                external = [true]
                r = 1
            end
            X = reshape(X, (1, length(X)))
        end

        if typeof(X) <: Matrix{T} where T<:AbstractFloat
            if r > 0
                if size(X)[1] != r
                    error("Invalid dimensions in X.")
                end
            else
                external = fill(true, size(X)[1])
                r = size(X)[1]
            end

        else
            error("Invalid specification of X.")
        end
    end

    if r > 0
        iI = findall(.!external)
        rI = length(iI)
        iE = findall(external)
        rE = length(iE)

        if model == "INGARCH"
            if link[1] == "Linear"
                if any(X .< 0)
                    error("Negative regressors invalid for linear link.")
                end
            end
        end

        if model == "INARMA"
            if link[1] == "Linear"
                if rI > 0
                    if any(X[iI, :] .< 0)
                        error("Negative regressors invalid for linear link.")
                    end
                end
            end

            if link[2] == "Linear"
                if rE > 0
                    if any(X[iE, :] .< 0)
                        error("Negative regressors invalid for linear link.")
                    end
                end
            end
        end
    end

    if sum(distr .== "NegativeBinomial") == 2
        if r == 0
            distr[2] = "Poisson"
        else
            if rE == 0
                distr[2] = "poisson"
            end
        end
    end

    if length(zi) != 1
        error("Invalid specification of zi.")
    end

    if typeof(zi) in [Array{Bool, 1}, Array{Bool, 2}]
        zi = zi[1]
    end
    if typeof(zi) != Bool
        error("Invalid specification of zi.")
    end

    if model == "INGARCH"
        if length(pastObs) + length(pastMean) == 0
            return IIDModel(distr[1], link[1], X, external, zi)
        end

        if length(pastMean) == 0
            return INARCHModel(distr[1], link[1], pastObs, X, external, zi)
        end

        if (length(pastObs) == 0) & (length(pastMean) > 0)
            error("INGARCH(0, q) not supported.")
        end
        return INGARCHModel(distr[1], link[1], pastObs, pastMean, X, external, zi)
    elseif model == "INARMA"
        if (length(pastMean) == 0) & (length(pastObs) > 0)
            return INARModel(distr, link, pastObs, X, external, zi)
        end

        if (length(pastMean) >= 0) & (length(pastObs) == 0)
            return INMAModel(distr, link, pastMean, X, external, zi)
        end

        return INARMAModel(distr, link, pastObs, pastMean, X, external, zi)
    else
        error("Invalid model framework chosen.")
    end
end
