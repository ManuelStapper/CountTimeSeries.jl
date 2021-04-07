import Base.show

function show(x::T where T<:CountModel)
    println("")
    println("Distribution: "*"\t\t"*reduce(*, x.distr.*" "))
    println("Link: "*"\t\t\t"*reduce(*, x.link.*" "))
    if typeof(x) != IIDModel
        if length(x.pastObs) > 0
            println("Past Obs: "*"\t\t"*reduce(*, string.(x.pastObs).*" "))
        end
    end

    if typeof(x) == INGARCHModel
        if length(x.pastMean) > 0
            println("Past Mean: "*"\t\t"*reduce(*, string.(x.pastMean).*" "))
        end
    end

    if length(x.X) > 0
        println("X: "*"\t\t\t"*string(size(x.X)))
    else
        println("X: "*"\t\t\t"*"None")
    end
    if length(x.external) > 0
        println("External: "*"\t\t"*reduce(*, ["N", "Y"][x.external .+ 1].*" "))
    end
    println("Zero Inflation: "*"\t"*ifelse(true, "Y", "N"))
end

function show(x::T where T<:INGARCHModel)
    println("")
    println("Distribution: "*"\t\t"*x.distr)
    println("Link: "*"\t\t\t"*x.link)

    if typeof(x) != IIDModel
        if length(x.pastObs) > 0
            println("Past Obs: "*"\t\t"*reduce(*, string.(x.pastObs).*" "))
        end
    end
    if typeof(x) == INGARCHModel
        if length(x.pastMean) > 0
            println("Past Mean: "*"\t\t"*reduce(*, string.(x.pastMean).*" "))
        end
    end
    if length(x.X) > 0
        println("X: "*"\t\t\t"*string(size(x.X)))
    else
        println("X: "*"\t\t\t"*"None")
    end
    if length(x.external) > 0
        println("External: "*"\t\t"*reduce(*, ["N", "Y"][x.external .+ 1].*" "))
    end
    println("Zero Inflation: "*"\t"*ifelse(true, "Y", "N"))
end

function show(x::T where T<:INARMA)
    println("")
    println("Distribution: "*"\t\t"*x.distr[1]*" and "*x.distr[2])
    println("Link: "*"\t\t\t"*x.link[1]*" and "*x.link[2])
    if typeof(x) != INMAModel
        if length(x.pastObs) > 0
            println("Past Obs: "*"\t\t"*reduce(*, string.(x.pastObs).*" "))
        end
    end
    if typeof(x) != INARModel
        if length(x.pastMean) > 0
            println("Past Mean: "*"\t\t"*reduce(*, string.(x.pastMean).*" "))
        end
    end
    if length(x.X) > 0
        println("X: "*"\t\t\t"*string(size(x.X)))
    else
        println("X: "*"\t\t\t"*"None")
    end
    if length(x.external) > 0
        println("External: "*"\t\t"*reduce(*, ["N", "Y"][x.external .+ 1].*" "))
    end
    println("Zero Inflation: "*"\t"*ifelse(true, "Y", "N"))
end

function show(x::parameter)
    println("")
    println("β0: "*"\t"*string(round(x.β0, digits = 4)))
    if length(x.α) > 0
        println("α: "*"\t"*reduce(*, string.(round.(x.α, digits = 4)).*" "))
    end

    if length(x.β) > 0
        println("β: "*"\t"*reduce(*, string.(round.(x.β, digits = 4)).*" "))
    end

    if length(x.η) > 0
        println("η: "*"\t"*reduce(*, string.(round.(x.η, digits = 4)).*" "))
    end

    if length(x.ϕ) > 0
        println("ϕ: "*"\t"*reduce(*, string.(round.(x.ϕ, digits = 4)).*" "))
    end

    if x.ω > 0
        println("ω: "*"\t"*reduce(*, string.(round.(x.ω, digits = 4)).*" "))
    end
end

function show(x::INGARCHresults)
    tStat = x.θ ./ x.se
    pVals = cdf.(Normal(), -abs.(tStat)).*2
    p1 = ifelse.(pVals .< 0.05, "*", "")
    p2 = ifelse.(pVals .< 0.01, "*", "")
    p3 = ifelse.(pVals .< 0.001, "*", "")
    stars = string.(p1, p2, p3)

    r = length(x.model.external)
    nb = x.model.distr == "NegativeBinomial"
    if typeof(x.model) == INGARCHModel
        name = ["β0"; string.("α", x.model.pastObs); string.("β", x.model.pastMean); string.("η", 1:r); ifelse(nb, "ϕ", []); ifelse(x.model.zi, "ω", [])]
    elseif typeof(x.model) == INARCHModel
        name = ["β0"; string.("α", x.model.pastObs); string.("η", 1:r); ifelse(nb, "ϕ", []); ifelse(x.model.zi, "ω", [])]
    else
        name = ["β0"; string.("η", 1:r); ifelse(nb, "ϕ", []); ifelse(x.model.zi, "ω", [])]
    end

    pr1 = round.(x.θ, digits = 4)
    pr2 = round.(x.se, digits = 4)
    pr3 = round.(pVals, digits = 4)
    pr4 = round.(x.CI[1, :], digits = 4)
    pr5 = round.(x.CI[2, :], digits = 4)
    println("")
    println("Results: Estimates, Standard Errors, p-values, Conf. Intervals")
    for i = 1:x.nPar
        println(name[i], "\t", pr1[i], "\t", pr2[i], "\t", pr3[i], "\t(", pr4[i], ", ", pr5[i], ")", stars[i])
    end
end

function show(x::INARMAresults)
    tStat = x.θ ./ x.se
    pVals = cdf.(Normal(), -abs.(tStat)).*2
    p1 = ifelse.(pVals .< 0.05, "*", "")
    p2 = ifelse.(pVals .< 0.01, "*", "")
    p3 = ifelse.(pVals .< 0.001, "*", "")
    stars = string.(p1, p2, p3)

    r = length(x.model.external)
    nb1 = x.model.distr[1] == "NegativeBinomial"
    nb2 = x.model.distr[2] == "NegativeBinomial"

    if typeof(x.model) == INARMAModel
        name = ["β0"; string.("α", x.model.pastObs); string.("β", x.model.pastMean); string.("η", 1:r); ifelse(nb1, "ϕ1", []); ifelse(nb2, "ϕ2", []); ifelse(x.model.zi, "ω", [])]
    elseif typeof(x.model) == INARModel
        name = ["β0"; string.("α", x.model.pastObs); string.("η", 1:r); ifelse(nb1, "ϕ1", []); ifelse(nb2, "ϕ2", []); ifelse(x.model.zi, "ω", [])]
    elseif typeof(x.model) == INMAModel
        name = ["β0"; string.("β", x.model.pastMean); string.("η", 1:r); ifelse(nb1, "ϕ1", []); ifelse(nb2, "ϕ2", []); ifelse(x.model.zi, "ω", [])]
    else
        name = ["β0"; string.("η", 1:r); ifelse(nb1, "ϕ1", []); ifelse(nb2, "ϕ2", []); ifelse(x.model.zi, "ω", [])]
    end

    pr1 = round.(x.θ, digits = 4)
    pr2 = round.(x.se, digits = 4)
    pr3 = round.(pVals, digits = 4)
    pr4 = round.(x.CI[1, :], digits = 4)
    pr5 = round.(x.CI[2, :], digits = 4)
    println("")
    println("Results: Estimates, Standard Errors, p-values, Conf. Intervals")
    for i = 1:x.nPar
        println(name[i], "\t", pr1[i], "\t", pr2[i], "\t", pr3[i], "\t(", pr4[i], ", ", pr5[i], ")", stars[i])
    end
end
