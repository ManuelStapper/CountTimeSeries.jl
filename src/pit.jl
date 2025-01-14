# Wrapper for PIT histogram functions

"""
    pit(results, nbins, level)
Function to compute the non-randomized PIT histogram, see [Czado et al.](https://mediatum.ub.tum.de/doc/1072660/1072660.pdf).

* `results`: Estimation results
* `nbins`: Number of bins (optional, default = 10)
* `level`: Confidence level (optional)

# Example
```julia-repl
pit(res, 10, 0.95)
```

If the argument `level` is put in, a confidence regio is added to the PIT histogram.
The height of all bins is inside that region if the PIT values follow a uniform distribution.
"""
function pit(results::INGARCHresults;
             nbins::Int64 = 10,
             level::Float64 = 0.0)
    y = results.y
    T = length(y)
    λ = results.λ
    zi = results.model.zi
    nb = results.model.distr == "NegativeBinomial"
    gp = results.model.distr == "GPoisson"

    if zi
        ω = results.pars.ω
        if nb
            ϕ = results.pars.ϕ[1]
            F = function(yt, λt)
                pt = ϕ/(ϕ + λt)
                d = MixtureModel([NegativeBinomial(ϕ, pt), Binomial(0)], [1 - ω, ω])
                cdf(d, yt)
            end
        elseif gp
            ϕ = results.pars.ϕ[1]
            F = function(yt, λt)
                d = MixtureModel([GPoisson(λt, ϕ), Binomial(0)], [1 - ω, ω])
                cdf(d, yt)
            end
        else
            F = function(yt, λt)
                d = MixtureModel([Poisson(λt), Binomial(0)], [1 - ω, ω])
                cdf(d, yt)
            end
        end
    else
        if nb
            ϕ = results.pars.ϕ[1]
            F = function(yt, λt)
                pt = ϕ/(ϕ + λt)
                d = NegativeBinomial(ϕ, pt)
                cdf(d, yt)
            end
        elseif gp
            ϕ = results.pars.ϕ[1]
            F = function(yt, λt)
                d = GPoisson(λt, ϕ)
                cdf(d, yt)
            end
        else
            F = function(yt, λt)
                d = Poisson(λt)
                cdf(d, yt)
            end
        end
    end

    u = collect(0:(1/nbins):1)
    mid = (u[2:end] .+ u[1:end - 1])./2
    vals = copy(u)

    for t = 1:T
        Px = F(y[t], λ[t])
        if y[t] == 0
            Px1 = 0
        else
            Px1 = F(y[t] - 1, λ[t])
        end
        if (Px - Px1) < 1e-16
            vals .+= (u .> Px)
        else
            vals .+= cdf.(Uniform(Px1, Px), u)
        end
    end

    heights = diff(vals)./T.*nbins
    # p = plot([0, 1], [1, 1], label = "", color = "red", linewidth = 2)

    if (level > 0) & (level < 1)
        cival = sqrt((nbins - 1)/T)*quantile(Normal(), (1 + level^(1/nbins))/2)
        # plot!([0, 1], fill(maximum([1 - cival, 0]), 2), color = "red", label = "", linestyle = :dash)
        # plot!([0, 1], fill(1 + cival, 2), color = "red", label = "", linestyle = :dash)
    end

    # bar!(mid, heights, bar_width = 1/nbins, alpha = 0.8, label = "", color = RGB(0, 105/255, 131/255))
    # title!("Non-Randomized PIT Histogram")
    # display(p)

    return mid, heights
end


function pit(results::INARMAresults{INARModel};
             nbins::Int64 = 10,
             level::Float64 = 0.0)
    y = results.y
    ymax = maximum(y)
    T = length(y)
    nb1 = results.model.distr[1] == "NegativeBinomial"
    nb2 = results.model.distr[2] == "NegativeBinomial"

    gp1 = results.model.distr[1] == "GPoisson"
    gp2 = results.model.distr[2] == "GPoisson"

    logl1 = results.model.link[1] == "Log"
    lin1 = !logl1
    logl2 = results.model.link[2] == "Log"
    lin2 = !logl2

    β0 = results.pars.β0
    p = length(results.model.pastObs)
    P = maximum(results.model.pastObs)

    r = length(results.model.external)

    α = results.pars.α
    if nb1 | gp1
        ϕ1 = results.pars.ϕ[1]
    end

    if nb2 | gp2
        ϕ2 = results.pars.ϕ[2]
    end

    zi = results.model.zi
    ω = results.pars.ω

    rE = sum(results.model.external)
    rI = sum(.!results.model.external)

    if r > 0
        iE = findall(results.model.external)
        iI = findall(.!results.model.external)

        η = results.pars.η
        ηI = η[iI]
        ηE = η[iE]
    end

    μ = zeros(T)
    λ = fill(β0, T)

    if rI > 0
        λ .+= η[iI]'*results.model.X[iI, :]
    end
    if logl1
        λ = exp.(λ)
    end

    if rE > 0
        μ = η[iE]'*results.model.X[iE, :]
        if logl2
            μ = exp.(μ)
        end
    end

    PR = zeros(ymax + 1, T)
    PZ = zeros(ymax + 1, T)

    for t = 1:T
        if nb1
            PR[:, t] = pdf.(NegativeBinomial(ϕ1, ϕ1/(ϕ1 + λ[t])), 0:ymax)
        elseif gp1
            PR[:, t] = pdf.(GPoisson(λ[t], ϕ1), 0:ymax)
        else
            PR[:, t] = pdf.(Poisson(λ[t]), 0:ymax)
        end

        if zi
            PR[2:ymax + 1, t] = PR[2:ymax + 1, t].*(1 .- ω./(1 .- PR[1, t]))
            PR[1, t] += ω
        end

        if rE > 0
            if nb2
                PZ[:, t] = pdf.(NegativeBinomial(ϕ2, ϕ2/(ϕ2 + μ[t])), 0:ymax)
            elseif gp2
                PZ[:, t] = pdf.(GPoisson(μ[t], ϕ2), 0:ymax)
            else
                PZ[:, t] = pdf.(Poisson(μ[t]), 0:ymax)
            end
        else
            PZ[:, t] = [1; fill(0, ymax)]
        end
    end

    PY = zeros(p, ymax + 1, ymax + 1)
    for i = 1:p
        for j = sort(unique(y))
            PY[i, j + 1, 1:j + 1] = pdf.(Binomial.(j, α[i]), 0:j)
        end
    end

    PAR = zeros(ymax + 1, P + 2)

    u = collect(0:(1/nbins):1)
    mid = (u[2:end] .+ u[1:end - 1])./2
    vals = copy(u)

    for t = P+1:T
        for i = 1:P
            PAR[:, i] = PY[i, y[t-i] + 1, :]
        end
        PAR[:, P + 1] = PZ[:, t]
        PAR[:, P + 2] = PR[:, t]

        ptemp = GetProbFromP(PAR)
        Px = sum(ptemp[1:y[t] + 1])
        Px1 = sum(ptemp[1:y[t]])

        if (Px - Px1) < 1e-16
            vals .+= (u .> Px)
        else
            vals .+= cdf.(Uniform(Px1, Px), u)
        end
    end

    heights = diff(vals)./T.*nbins
    # p = plot([0, 1], [1, 1], label = "", color = "red", linewidth = 2)

    if (level > 0) & (level < 1)
        cival = sqrt((nbins - 1)/T)*quantile(Normal(), (1 + level^(1/nbins))/2)
        # plot!([0, 1], fill(maximum([0, 1 - cival]), 2), color = "red", label = "", linestyle = :dash)
        # plot!([0, 1], fill(1 + cival, 2), color = "red", label = "", linestyle = :dash)
    end

    # bar!(mid, heights, bar_width = 1/nbins, alpha = 0.8, label = "", color = RGB(0, 105/255, 131/255))
    # title!("Non-Randomized PIT Histogram")
    # display(p)

    return mid, heights
end
