module Domains
using LinearAlgebra
@inline function square(I::CartesianIndex , Ids::CartesianIndices)::Float32
    Id = oneunit(I)
    if I in 2*(Ids[begin]+Id):2*(Ids[end]-Id)
        return 1
    end
    return 0
    end
@inline function circle(I::CartesianIndex , Ids::CartesianIndices)::Float32
    @inline r  = Ids[end] - I
    m = maximum(Tuple(Ids[end]))
    if norm(Tuple(r)) < 0.8 * m
        return 1.
        end
    return 0.
    end

end
