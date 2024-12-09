using CountTimeSeries
using Test

@testset "CountTimeSeries.jl" begin
    using Random, Distributions
    X = reshape(collect(1:100) .+ 0.0, (1, 100))
    X2 = vcat((1:100)', ((1:100).^2)') .+ 0.0
    models = Array{Any, 1}(undef, 30)

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

    models[11] = Model(pastObs = 1, pastMean = 1, X = X2, external = [true, false], zi = true)
    models[12] = Model(pastObs = 1, X = X2, external = [true, false], zi = true)
    models[13] = Model(model = "INARMA", pastObs = 1, pastMean = 1, X = X2, external = [true, false], zi = true)
    models[14] = Model(model = "INARMA", pastObs = 1, X = X2, external = [true, false], zi = true)
    models[15] = Model(model = "INARMA", pastMean = 1, X = X2, external = [true, false], zi = true)
    models[16] = Model(pastObs = 1, pastMean = 1, distr = "NegativeBinomial", X = X2, external = [true, false], zi = true)
    models[17] = Model(pastObs = 1, distr = "NegativeBinomial", X = X2, external = [true, false], zi = true)
    models[18] = Model(model = "INARMA", pastObs = 1, pastMean = 1, distr = ["NegativeBinomial", "Poisson"], X = X2, external = [true, false], zi = true)
    models[19] = Model(model = "INARMA", pastObs = 1, distr = "NegativeBinomial", X = X2, external = [true, false], zi = true)
    models[20] = Model(model = "INARMA", pastMean = 1, distr = "NegativeBinomial", X = X2, external = [true, false], zi = true)

    models[21] = Model(pastObs = 1, pastMean = 1, X = X, external = [true], zi = true, link = "Log")
    models[22] = Model(pastObs = 1, X = X, external = [true], zi = true, link = "Log")
    models[23] = Model(model = "INARMA", pastObs = 1, pastMean = 1, X = X, external = [true], zi = true, link = "Log")
    models[24] = Model(model = "INARMA", pastObs = 1, X = X, external = [true], zi = true, link = "Log")
    models[25] = Model(model = "INARMA", pastMean = 1, X = X, external = [true], zi = true, link = "Log")
    models[26] = Model(pastObs = 1, pastMean = 1, distr = "NegativeBinomial", X = X, external = [true], zi = true, link = "Log")
    models[27] = Model(pastObs = 1, distr = "NegativeBinomial", X = X, external = [true], zi = true, link = "Log")
    models[28] = Model(model = "INARMA", pastObs = 1, pastMean = 1, distr = ["NegativeBinomial", "Poisson"], X = X, external = [true], zi = true, link = "Log")
    models[29] = Model(model = "INARMA", pastObs = 1, distr = "NegativeBinomial", X = X, external = [true], zi = true, link = "Log")
    models[30] = Model(model = "INARMA", pastMean = 1, distr = "NegativeBinomial", X = X, external = [true], zi = true, link = "Log")


    pars = Array{parameter, 1}(undef, 30)
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

    pars[11] = θ2par([10, 0.5, 0.2, 0.1, 0.001, 0.1], models[11])
    pars[12] = θ2par([10, 0.5, 0.1, 0.001, 0.1], models[12])
    pars[13] = θ2par([10, 0.5, 0.2, 0.1, 0.001, 0.1], models[13])
    pars[14] = θ2par([10, 0.5, 0.1, 0.001, 0.1], models[14])
    pars[15] = θ2par([10, 0.2, 0.1, 0.001, 0.1], models[15])
    pars[16] = θ2par([10, 0.5, 0.2, 0.1, 0.001, 3, 0.1], models[16])
    pars[17] = θ2par([10, 0.5, 0.1, 0.001, 3, 0.1], models[17])
    pars[18] = θ2par([10, 0.5, 0.2, 0.1, 0.001, 3, 0.1], models[18])
    pars[19] = θ2par([10, 0.2, 0.1, 0.001, 3, 3, 0.1], models[19])
    pars[20] = θ2par([10, 0.5, 0.1, 0.001, 3, 3, 0.1], models[20])

    pars[21] = θ2par([1, 0.5, 0.2, -0.1, 0.1], models[21])
    pars[22] = θ2par([1, 0.5, -0.1, 0.1], models[22])
    pars[23] = θ2par([1, 0.5, 0.2, -0.1, 0.1], models[23])
    pars[24] = θ2par([1, 0.5, -0.1, 0.1], models[24])
    pars[25] = θ2par([1, 0.2, -0.1, 0.1], models[25])
    pars[26] = θ2par([1, 0.5, 0.2, -0.1, 3, 0.1], models[26])
    pars[27] = θ2par([1, 0.5, -0.1, 3, 0.1], models[27])
    pars[28] = θ2par([1, 0.5, 0.2, -0.1, 3, 0.1], models[28])
    pars[29] = θ2par([1, 0.2, -0.1, 3, 3, 0.1], models[29])
    pars[30] = θ2par([1, 0.5, -0.1, 3, 3, 0.1], models[30])

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
    predict(res[2], 10, Xnew)
    pit(res[2])

    convert(INGARCHModel, models[2])
    convert(INGARCHModel, Model())
    convert(INARCHModel, Model())
    convert(INARMAModel, models[4])

    show.(models)
    show.(res)
    show.(pars)

    model = Model(model = "INARMA", pastMean = 1:2)
    y = simulate(100, model, [10, 0.5, 0.2])[1]
    ll(y, model, [10, 0.5, 0.2])

    model = Model(model = "INARMA", pastMean = 1:2, pastObs = 1, zi = true)
    y = simulate(100, model, [10, 0.1, 0.5, 0.2, 0.1])[1]
    ll(y, model, [10, 0.1, 0.5, 0.2, 0.1])

    model = Model(zi = true)
    y = simulate(100, model, [10.0, 0.1])[1]
    ll(y, model, [10, 0.1])

    model = Model(zi = true, X = X2, external = [true, true])
    y = simulate(100, model, [10, 0.1, 0.01, 0.01])[1]
    ll(y, model, [10, 0.1, 0.01, 0.01])

    model = Model(model = "INARMA", pastObs = 1)
    pars = θ2par([10, 0.5], model)
    y = simulate(100, model, pars)[1]
    CountTimeSeries.fittedValues(y, model, pars)

    model = Model(model = "INARMA", pastObs = 1, pastMean = 1)
    pars = θ2par([10, 0.5, 0.5], model)
    y = simulate(100, model, pars)[1]
    CountTimeSeries.fittedValues(y, model, pars)

    mean(model, pars)
    var(model, pars)
    acf(model, pars, 10)
    acvf(model, pars, 10)

    model = Model(model = "INGARCH", pastObs = 1, pastMean = 1)
    pars = θ2par([10, 0.5, 0.3], model)
    mean(model, pars)
    mean(model, pars, true)
    var(model, pars)
    var(model, pars, true)
    acf(model, pars, 10)
    acf(model, pars, 10, true)
    acvf(model, pars, 10)
    acvf(model, pars, 10, true)

    0.5 ∘ 10
    Poisson(0.5) ∘ 10
    0.5 ∘ Poisson(10)
    Binomial(1, 0.5) ∘ Poisson(10)
    u = rand(1)[1]
    ∘(0.5, 10, u)
    ∘(Binomial(1, 0.5), 10, rand(10))


    model = Model(model = "INGARCH", distr = "NegativeBinomial", pastObs = 1, pastMean = 1)
    y = CountTimeSeries.simulate(100, model, [10, 0.5, 0.2, 3])[1]
    res = fit(y, model)
    pit(res)

    model = Model(model = "INGARCH", distr = "NegativeBinomial", pastObs = 1, pastMean = 1, zi = true)
    y = CountTimeSeries.simulate(100, model, [10, 0.5, 0.2, 3, 0.2])[1]
    res = fit(y, model)
    pit(res)

    model = Model(model = "INARMA", distr = "NegativeBinomial", pastObs = 1)
    y = CountTimeSeries.simulate(100, model, [10, 0.5, 3])[1]
    res = fit(y, model)
    pit(res)

    model = Model(model = "INARMA",  pastObs = 1)
    y = CountTimeSeries.simulate(100, model, [10, 0.5])[1]
    res = fit(y, model)
    pit(res)
end