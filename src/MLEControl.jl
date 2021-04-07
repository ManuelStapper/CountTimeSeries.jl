# Settings for fitting

"""
    MLEControl(init::parameter, optimizer::String, ci::Bool, maxEval<:Integer)
Structure for esitmation settings.

* `init`: Initial values for optimization
* `optimizer`: Optimizing Routine, "NelderMead", "BFGS" or "LBFGS"
* `ci`: Indicator: Shall confidence intervals be computed?
* `maxEval`: Maximum number of likelihood evaluations
"""
mutable struct MLEControl
    init::parameter
    optimizer::String
    ci::Bool
    maxEval::T where T<:Integer
end
