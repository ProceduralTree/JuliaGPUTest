using LinearAlgebra
@inline function G(I::CartesianIndex , Ids::CartesianIndices)::Float32
    @inline r  = Ids[end] - I
    m = max(Tuple(Ids[end]))
    if norm(Tuple(r)) < 1.8 * m
        return 1
        end
    return 0
    end
