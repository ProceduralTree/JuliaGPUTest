
@inline function G(I::CartesianIndex, Inds::CartesianIndices)::Float32
    Id = oneunit(I)
    if I in 2*(Inds[begin]+Id):2*(Inds[end]-Id)
        return 1
    end
    return 0
end
