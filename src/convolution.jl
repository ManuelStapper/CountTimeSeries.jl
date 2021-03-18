# Convolute two vectors of probabilities

# Input:
# p1:       Vector of probabilities P(X = 0), ..., P(X = n)
# p2:       Vector of probabilities P(Y = 0), ..., P(Y = n)

# Output:
# out:      Vector of probabilities P(X + Y = 0), ..., P(X + Y = n)

function convolution(p1::Vector{Float64},
                     p2::Vector{Float64})::Vector{Float64}
    #
    M = length(p1)
    out = zeros(M)
    @inbounds for i = 0:M-1
        for j = 0:M-1-i
            out[i+j+1] += p1[i+1]*p2[j+1]
        end
    end
    out
end

function convolution(p1::Vector{Float64},
                     p2::Vector{Float64},
                     x::Int64)::Vector{Float64}
    #
    out = zeros(x+1)
    @inbounds for i = 0:x
        for j = 0:x-i
            out[i+j+1] += p1[i+1]*p2[j+1]
        end
    end
    out
end

function convolutionRev(p1::Vector{Float64},
                     p2::Vector{Float64},
                     x::Int64)::Vector{Float64}
    #
    out = zeros(x+1)
    @inbounds for i = 0:x
        for j = 0:x-i
            out[x + 1 - i - j] += p1[i+1]*p2[j+1]
        end
    end
    out
end

function convolutionRev(p1::Vector{Float64},
                     p2::Matrix{Float64},
                     x::Int64)::Matrix{Float64}
    #
    ncol = size(p2)[2]
    out = zeros(x+1, ncol)
    @inbounds for i = 0:x
        for j = 0:x-i
            for col = 1:ncol
                out[x + 1 - i - j, col] += p1[i+1]*p2[j+1, col]
            end
        end
    end
    out
end

function convolutionRevScale(p1::Vector{Float64},
                             p2::Matrix{Float64},
                             x::Int64,
                             scale::Vector{Float64})::Matrix{Float64}
    ncol = size(p2)[2]
    out = zeros(ncol, x+1)
    @inbounds for i = 0:x
        for j = 0:x-i
            scalefactor = scale[x + 1 - i - j]
            p1Part = p1[i+1]
            for col = 1:ncol
                out[col, x + 1 - i - j] += p1Part*p2[j+1, col]*scalefactor
            end
        end
    end
    out
end
