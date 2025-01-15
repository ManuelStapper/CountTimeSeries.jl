# CountTimeSeries

[![Build status](https://ci.appveyor.com/api/projects/status/frnihr2qw4328rnf?svg=true)](https://ci.appveyor.com/project/ManuelStapper/counttimeseries-jl-xqtaf)
[![Coverage](https://codecov.io/gh/ManuelStapper/CountTimeSeries.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/ManuelStapper/CountTimeSeries.jl)
[![DOI](https://zenodo.org/badge/349195207.svg)](https://zenodo.org/badge/latestdoi/349195207)


This package is developed to handle univariate count data time series. Up to now, it covers integer counterparts of ARMA and GARCH processes with broad generalizations. It enables the user to generate artificial data, estimate parameters by Maximum Likelihood, conduct inference on the estimates, assess model choice and carry out forecasts.

## Example

A model, for example a simple INGARCH(1, 1) with Poisson distribution is defined first by
```julia
model = Model(pastObs = 1, pastMean = 1)
```
Then, a time series is simulated by
```julia
y = simulate(1000, model, [10, 0.5, 0.2])[1]
```
its parameters estimates
```julia
res = fit(y, model)
```
and finally a 10-step ahead prediction carried out
```julia
predict(res, 10)
```

## Future Extensions
* Bounded Counts
* GMM Estimation
* Multivariate Processes
