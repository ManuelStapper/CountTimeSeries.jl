# Add-on function for Quasi Poisson estimation of INAGRCH models

"""
    QPois(results)
Add-on function for Quasi Poisson estimation of INGARCH models.

* `results`: Estimation results (only INGARCH)

# Example
```julia-repl
QPois(results)
```

The function uses estimation results of an INGARCH fit with Poisson distribution
and estimates the overdispersion parameter according to
[Christou and Fokianos (2013)](https://doi.org/10.1111/jtsa.12050).
The function puts out a changed version of the input including an estimate of
the overdispersion parameter. The distribution is thereby changed to "NegativeBinomial".
"""
function QPois(results::INGARCHresults)
    if results.model.distr != "Poisson"
        error("Only results of Poisson INGARCH-fit supported.")
    end

    function rootFun(ϕrf, results)
        y = results.y
        T = length(y)

        λ = results.λ
        if typeof(results.model) != IIDModel
            po = results.model.pastObs
        else
            po = Array{Int64, 1}([])
        end

        if typeof(results.model) == INGARCHModel
            pm = results.model.pastMean
        else
            pm = Array{Int64, 1}([])
        end

        if length(po) == 0
            P = 0
            p = 0
        else
            P = maximum(po)
            p = length(po)
        end

        if length(pm) == 0
            Q = 0
            q = 0
        else
            Q = maximum(pm)
            q = length(pm)
        end

        M = maximum([P, Q])

        r = length(results.model.external)

        indrel = M+1:T
        λrf = λ[indrel]
        sum((y[indrel] .- λrf).^2 ./ (λrf .+ λrf.^2 ./ exp(ϕrf))) - T - p - q - r - 1
    end

    if sign(rootFun(-100, results)) == sign(rootFun(10000, results))
        println("Root search failed. Overdispersion parameter is set to 10000")
        ϕest = 10000.0
    else
        ϕest = exp(find_zero(x -> rootFun(x, results), 10))
    end

    # Rewrite results
    out = INGARCHresults(results.y, results.θ, results.pars, results.λ, results.residuals,
                         results.LL, results.LLs, results.nPar, results.nObs, results.se,
                         results.CI, results.model, results.converged, results.MLEControl)

    out.pars.ϕ = [ϕest]
    out.model.distr = "NegativeBinomial"
    out.θ = par2θ(out.pars, out.model)

    return out
end
