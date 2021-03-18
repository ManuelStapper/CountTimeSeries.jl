# For vectors of probabilities in columns of P compute the convolution

# Input:
# P:        Matrix P_{ij} of probabilities P(X_j = i - 1)
#           for i = 1, ..., n and j = 1, ..., k

# Output:
# p:        Vector of probabilities P(X_1 + ... + X_k = i)
#           for i = 0, ..., n

function GetProbFromP(P::Array{Float64, 2})::Vector{Float64}
    dims = size(P)[2]
    
    if dims == 1
        return P[:, 1]
    end

    if dims > 2
        for i = dims - 1:-1:2
            P[:, i] = convolution(P[:, i], P[:, i+1])
        end
    end

    convolution(P[:, 1], P[:, 2])
end
