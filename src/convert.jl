import Base.convert

function convert(::Type{INARCHModel}, model::INGARCHModel)
    INARCHModel(model.distr, model.link, model.pastObs, model.X, model.external, model.zi)
end

function convert(::Type{INGARCHModel}, model::INARCHModel)
    INGARCHModel(model.distr, model.link, model.pastObs, Vector{Int64}([]), model.X, model.external, model.zi)
end

function convert(::Type{INGARCHModel}, model::IIDModel)
    INGARCHModel(model.distr, model.link, Vector{Int64}([]), Vector{Int64}([]), model.X, model.external, model.zi)
end

function convert(::Type{IIDModel}, model::INGARCHModel)
    if length(model.pastObs) + length(pastMean) > 0
        error("Can not convert model to IIDModel.")
    end
    IIDModel(model.distr, model.link, model.X, model.external, model.zi)
end

function convert(::Type{INARCHModel}, model::IIDModel)
    INARCHModel(model.distr, model.link, Vector{Int64}([]), model.X, model.external, model.zi)
end

function convert(::Type{IIDModel}, model::INARCHModel)
    if length(model.pastObs) > 0
        error("Can not convert model to IIDModel.")
    end
    IIDModel(model.distr, model.link, model.X, model.external, model.zi)
end

function convert(::Type{INARMAModel}, model::INARModel)
    INARMAModel(model.distr, model.link, model.pastObs, Vector{Int64}([]), model.X, model.external, model.zi)
end

function convert(::Type{INARModel}, model::INARMAModel)
    if length(model.pastMean) > 0
        error("Can not convert model to INARModel")
    end
    INARModel(model.distr, model.link, model.pastObs, model.X, model.external, model.zi)
end

function convert(::Type{INARMAModel}, model::INMAModel)
    INARMAModel(model.distr, model.link, Vector{Int64}([]), model.pastMean, model.X, model.external, model.zi)
end

function convert(::Type{INMAModel}, model::INARMAModel)
    if length(model.pastObs) > 0
        error("Can not convert model to INMAModel.")
    end
    INMAModel(model.distr, model.link, model.pastMean, model.X, model.external, model.zi)
end
