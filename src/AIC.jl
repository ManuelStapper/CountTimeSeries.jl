# Functions for information criteria

# Input:
# results:      Results of estimation

# Output:
# Information criterion

"""
    AIC(results, dropfirst)
Computing the Akaike information criterion.

* `results`: Result from fitting a count data model.
* `dropfirst`: Can be used to exclude the first observations from computation.

# Examples
```julia-repl
AIC(res1, 2) # res1: Results from INARCH(1) fit
AIC(res2)    # res2: Results from INARCH(2) fit
```
"""
function AIC(results::INGARCHresults, dropfirst::T where T<:Integer = -1)
    if dropfirst == -1
        LLmax = results.LL
    else
        LLmax = sum(results.LLs[dropfirst + 1:end])
    end

    nPar = results.nPar

    -2*LLmax + 2*nPar
end

function AIC(results::INARMAresults, dropfirst::T where T<:Integer = -1)
    if dropfirst == -1
        LLmax = results.LL
    else
        LLmax = sum(results.LLs[dropfirst + 1:end])
    end

    nPar = results.nPar

    -2*LLmax + 2*nPar
end
