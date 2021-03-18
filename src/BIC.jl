"""
    BIC(results::Rspec, dropfirst::Int64)
Computing the Bayes information criterion.

* `results`: Result from fitting a count data model.
* `dropfirst`: Can be used to exclude the first observations from computation.

# Examples
```julia-repl
BIC(res1, 2) # res1: Results from INARCH(1) fit
BIC(res2)    # res2: Results from INARCH(2) fit
```
"""
function BIC(results::INGARCHresults, dropfirst::T where T<:Integer = -1)
    if dropfirst == -1
        LLmax = results.LL
        nObs = results.nObs
    else
        LLmax = sum(results.LLs[dropfirst + 1:end])
        nObs = length(results.y) - dropfirst
    end
    nPar = results.nPar

    -2*LLmax + log(nObs)*nPar
end

function BIC(results::INARMAresults, dropfirst::T where T<:Integer = -1)
    if dropfirst == -1
        LLmax = results.LL
        nObs = results.nObs
    else
        LLmax = sum(results.LLs[dropfirst + 1:end])
        nObs = length(results.y) - dropfirst
    end
    nPar = results.nPar

    -2*LLmax + log(nObs)*nPar
end
