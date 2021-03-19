using CountTimeSeries
using Test

using Random

@testset "CountTimeSeries.jl" begin
    X = reshape(collect(1:100) .+ 0.0, (1, 100))
    model1 = Model(pastObs = 1, pastMean = 1, X = X, external = [true], zi = true)
    model2 = Model(pastObs = 1, X = X, external = [true], zi = true)
    model3 = Model(model = "INARMA", pastObs = 1, pastMean = 1, X = X, external = [true], zi = true)
    model4 = Model(model = "INARMA", pastObs = 1, X = X, external = [true], zi = true)
    model5 = Model(model = "INARMA", pastMean = 1, X = X, external = [true], zi = true)

    model6 = Model(pastObs = 1, pastMean = 1, distr = "NegativeBinomial", X = X, external = [true], zi = true)
    model7 = Model(pastObs = 1, distr = "NegativeBinomial", X = X, external = [true], zi = true)
    model8 = Model(model = "INARMA", pastObs = 1, pastMean = 1, distr = ["NegativeBinomial", "Poisson"], X = X, external = [true], zi = true)
    model9 = Model(model = "INARMA", pastObs = 1, distr = "NegativeBinomial", X = X, external = [true], zi = true)
    model10 = Model(model = "INARMA", pastMean = 1, distr = "NegativeBinomial", X = X, external = [true], zi = true)

    Random.seed!(1)
    y1 = simulate(100, model1, [10, 0.5, 0.2, 0.1, 0.1])[1]
    y2 = simulate(100, model2, [10, 0.5, 0.1, 0.1])[1]
    y3 = simulate(100, model3, [10, 0.5, 0.2, 0.1, 0.1])[1]
    y4 = simulate(100, model4, [10, 0.5, 0.1, 0.1])[1]
    y5 = simulate(100, model5, [10, 0.2, 0.1, 0.1])[1]

    y6 = simulate(100, model6, [10, 0.5, 0.2, 0.1, 3, 0.1])[1]
    y7 = simulate(100, model7, [10, 0.5, 0.1, 3, 0.1])[1]
    y8 = simulate(100, model8, [10, 0.5, 0.2, 0.1, 3, 0.1])[1]
    y9 = simulate(100, model9, [10, 0.2, 0.1, 3, 3, 0.1])[1]
    y10 = simulate(100, model10, [10, 0.5, 0.1, 3, 3, 0.1])[1]

    ll(y1, model1, [10, 0.5, 0.2, 0.1, 0.1])
    ll(y2, model2, [10, 0.5, 0.1, 0.1])
    ll(y3, model3, [10, 0.5, 0.2, 0.1, 0.1])
    ll(y4, model4, [10, 0.5, 0.1, 0.1])
    ll(y5, model5, [10, 0.2, 0.1, 0.1])

    ll(y6, model6, [10, 0.5, 0.2, 0.1, 3, 0.1])
    ll(y7, model7, [10, 0.5, 0.1, 3, 0.1])
    ll(y8, model8, [10, 0.5, 0.2, 0.1, 3, 0.1])
    ll(y9, model9, [10, 0.2, 0.1, 3, 3, 0.1])
    ll(y10, model10, [10, 0.5, 0.1, 3, 3, 0.1])

    setting = MLESettings(y1, model1, [10, 0.5, 0.2, 0.1, 0.1], ci = true)
    res1 = fit(y1, model1, setting, printResults = false)
    fit(y2, model2, printResults = false)
    fit(y4, model4, printResults = false)

    fit(y6, model6, printResults = false)
    fit(y7, model7, printResults = false)
    fit(y9, model9, printResults = false)

    res61 = fit(y6, model1, printResults = false)
    QPois(res61)

    AIC(res1)
    BIC(res1)
    HQIC(res1)
    Xnew = reshape(collect(101:110) .+ 0.0, (1, 10))
    predict(res1, 10, Xnew)
    predict(res1, 10, 100, Xnew)
end
