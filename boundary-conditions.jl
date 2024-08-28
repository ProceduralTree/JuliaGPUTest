module BoundaryKernels
using KernelAbstractions

function G(I::CartesianIndex, Inds::CartesianIndices)
    Id = oneunit(I)
    if I in 2*(Inds[begin]+Id):2*(Inds[end]-Id)
        return 1
    end
    return 0
end


@kernel function constant(out)
    I = @index(Global, Cartesian)
    Ids = CartesianIndices(Φ)
    Id = oneunit(I)
    Ix = CartesianIndex(1,0)
    Iy = CartesianIndex(0,1)
    value = 1.
    out[I] = 0.0
    if I in Ids[begin]+Id:Ids[end]-Id
        out[I] = value * abs(
            G(2 * I + Ix, Ids)
            + G(2 * I + Iy, Ids)
            + G(2 * I - Ix, Ids)
            + G(2 * I - Iy, Ids)
            - 4* G(2 * I, Ids))
    end
end

@kernel function left(out)
    I = @index(Global, Cartesian)
    Ids = CartesianIndices(out)
    Id = oneunit(I)
    Ix = CartesianIndex(1,0)
    Iy = CartesianIndex(0,1)
    value = 1.
    out[I] = 0.0
    if I in Ids[begin]+Id:Ids[end]-Id
        out[I] = value * abs(G(2 * I - Ix, Ids) -  G(2 * I, Ids))
    end
end

@kernel function right(out)
    I = @index(Global, Cartesian)
    Ids = CartesianIndices(out)
    Id = oneunit(I)
    Ix = CartesianIndex(1,0)
    Iy = CartesianIndex(0,1)
    value = 1.
    out[I] = 0.0
    if I in Ids[begin]+Id:Ids[end]-Id
        out[I] = value * abs(G(2 * I + Ix, Ids) -  G(2 * I, Ids))
    end
end
@kernel function top(out)
    I = @index(Global, Cartesian)
    Ids = CartesianIndices(out)
    Id = oneunit(I)
    Ix = CartesianIndex(1,0)
    Iy = CartesianIndex(0,1)
    value = 1.
    out[I] = 0.0
    if I in Ids[begin]+Id:Ids[end]-Id
        out[I] = value * abs(G(2 * I + Iy, Ids) -  G(2 * I, Ids))
    end
end
@kernel function bottom(out)
    I = @index(Global, Cartesian)
    Ids = CartesianIndices(out)
    Id = oneunit(I)
    Ix = CartesianIndex(1,0)
    Iy = CartesianIndex(0,1)
    value = 1.
    out[I] = 0.0
    if I in Ids[begin]+Id:Ids[end]-Id
        out[I] = value * abs(G(2 * I - Iy, Ids) -  G(2 * I, Ids))
    end
end
end
