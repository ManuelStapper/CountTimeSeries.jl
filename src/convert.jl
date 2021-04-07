import Base.convert

function convert(::Type{INARCHModel}, x::INGARCHModel)
    INARCHModel(x.distr, x.link, x.pastObs, x.X, x.external, x.zi)
end

function convert(::Type{INGARCHModel}, x::INARCHModel)
    INGARCHModel(x.distr, x.link, x.pastObs, Array{Int64, 1}([]), x.X, x.external, x.zi)
end

function convert(::Type{INGARCHModel}, x::IIDModel)
    INGARCHModel(x.distr, x.link, Array{Int64, 1}([]), Array{Int64, 1}([]), x.X, x.external, x.zi)
end

function convert(::Type{IIDModel}, x::INGARCHModel)
    if length(x.pastObs) + length(pastMean) > 0
        error("Can not convert model to IIDModel.")
    end
    IIDModel(x.distr, x.link, x.X, x.external, x.zi)
end

function convert(::Type{INARCHModel}, x::IIDModel)
    INARCHModel(x.distr, x.link, Array{Int64, 1}([]), x.X, x.external, x.zi)
end

function convert(::Type{IIDModel}, x::INARCHModel)
    if length(x.pastObs) > 0
        error("Can not convert model to IIDModel.")
    end
    IIDModel(x.distr, x.link, x.X, x.external, x.zi)
end

function convert(::Type{INARMAModel}, x::INARModel)
    INARMAModel(x.distr, x.link, x.pastObs, Array{Int64, 1}([]), x.X, x.external, x.zi)
end

function convert(::Type{INARModel}, x::INARMAModel)
    if length(x.pastMean) > 0
        error("Can not convert model to INARModel")
    end
    INARModel(x.distr, x.link, x.pastObs, x.X, x.external, x.zi)
end

function convert(::Type{INARMAModel}, x::INMAModel)
    INARMAModel(x.distr, x.link, Array{Int64, 1}([]), x.pastMean, x.X, x.external, x.zi)
end

function convert(::Type{INMAModel}, x::INARMAModel)
    if length(x.pastObs) > 0
        error("Can not convert model to INMAModel.")
    end
    INMAModel(x.distr, x.link, x.pastMean, x.X, x.external, x.zi)
end
