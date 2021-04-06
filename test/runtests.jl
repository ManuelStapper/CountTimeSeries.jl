using CountTimeSeries
using Test

@testset "CountTimeSeries.jl" begin
    using Random
    X = reshape(collect(1:100) .+ 0.0, (1, 100))
    models = Array{Any, 1}(undef, 10)

    models[1] = Model(pastObs = 1, pastMean = 1, X = X, external = [true], zi = true)
    models[2] = Model(pastObs = 1, X = X, external = [true], zi = true)
    models[3] = Model(model = "INARMA", pastObs = 1, pastMean = 1, X = X, external = [true], zi = true)
    models[4] = Model(model = "INARMA", pastObs = 1, X = X, external = [true], zi = true)
    models[5] = Model(model = "INARMA", pastMean = 1, X = X, external = [true], zi = true)
    models[6] = Model(pastObs = 1, pastMean = 1, distr = "NegativeBinomial", X = X, external = [true], zi = true)
    models[7] = Model(pastObs = 1, distr = "NegativeBinomial", X = X, external = [true], zi = true)
    models[8] = Model(model = "INARMA", pastObs = 1, pastMean = 1, distr = ["NegativeBinomial", "Poisson"], X = X, external = [true], zi = true)
    models[9] = Model(model = "INARMA", pastObs = 1, distr = "NegativeBinomial", X = X, external = [true], zi = true)
    models[10] = Model(model = "INARMA", pastMean = 1, distr = "NegativeBinomial", X = X, external = [true], zi = true)

    pars = Array{parameter, 1}(undef, 10)
    pars[1] = θ2par([10, 0.5, 0.2, 0.1, 0.1], models[1])
    pars[2] = θ2par([10, 0.5, 0.1, 0.1], models[2])
    pars[3] = θ2par([10, 0.5, 0.2, 0.1, 0.1], models[3])
    pars[4] = θ2par([10, 0.5, 0.1, 0.1], models[4])
    pars[5] = θ2par([10, 0.2, 0.1, 0.1], models[5])
    pars[6] = θ2par([10, 0.5, 0.2, 0.1, 3, 0.1], models[6])
    pars[7] = θ2par([10, 0.5, 0.1, 3, 0.1], models[7])
    pars[8] = θ2par([10, 0.5, 0.2, 0.1, 3, 0.1], models[8])
    pars[9] = θ2par([10, 0.2, 0.1, 3, 3, 0.1], models[9])
    pars[10] = θ2par([10, 0.5, 0.1, 3, 3, 0.1], models[10])

    Random.seed!(1)
    ys = (x -> x[1]).(simulate.(100, models, pars))
    LLs = (x -> x[1]).(ll.(ys, models, pars))

    settings = MLESettings.(ys, models, pars, ci = false)
    res = fit.(ys[[1, 2, 4, 6, 7, 9]], models[[1, 2, 4, 6, 7, 9]], settings[[1, 2, 4, 6, 7, 9]], printResults = true)

    res61 = fit(ys[6], models[1], printResults = false)
    res61qpois = QPois(res61)
    models[1] = Model(pastObs = 1, pastMean = 1, X = X, external = [true], zi = true)

    AIC.(res)
    BIC.(res)
    HQIC.(res)
    Xnew = reshape(collect(101:110) .+ 0.0, (1, 10))

    for i = 2:6
        predict(res[i], 10, 100, Xnew)
    end
    pit(res[2])
end
