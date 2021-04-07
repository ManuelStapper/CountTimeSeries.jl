## List of types, structs and functions
### Types

```@docs
CountModel
```

```@docs
INGARCH
```

```@docs
INARMA
```

### Structs

```@docs
INGARCHModel
```

```@docs
INARCHModel
```

```@docs
IIDModel
```

```@docs
INARMAModel
```

```@docs
INARModel
```

```@docs
INMAModel
```

```@docs
parameter
```

### Functions

```@docs
AIC(results, dropfirst)
```

```@docs
BIC(results, dropfirst)
```

```@docs
HQIC(results, dropfirst)
```

```@docs
fit(y, model, MLEControl, printResults = true, initiate = "first")
```

```@docs
ll(y, model, θ; initiate = "first")
```

```@docs
MLESettings(y, model, init, optimizer, ci)
```

```@docs
Model(model, distr, link, pastObs, pastMean, X,
    external, zi)
```

```@docs
par2θ(θ, model)
```

```@docs
pit(results, nbins, level)
```

```@docs
predict(results, h, nChain, Xnew)
```

```@docs
QPois(results)
```

```@docs
simulate(T, model, θ; burnin, pinfirst)
```

```@docs
θ2par(θ, model)
```

```@docs
∘
```
