using KernelAbstractions
using LinearAlgebra


@kernel  function domain(out)
    I = @index(Global, Cartesian)
    Inds = CartesianIndices(out)
    @inline out[I] = G( 2 * I , Inds ) * out[I]
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


"""
# border(out)
-----------------
* returns: all grid-cells on the border
* out: Array to be written to
# This function is a kernel abstraction.
plase initialize it first on a dedicated device eg

julia> using KernelAbstractions
julia> using oneAPI
julia> array = oneArray(zeros(10,10))
julia> b = BoundaryKernels.border(get_backend(array) , 128 , size(array))
julia> b(array)

"""
@kernel function border(out)
    I = @index(Global, Cartesian)
    Ids = CartesianIndices(out)
    Id = oneunit(I)
    Ix = CartesianIndex(1,0)
    Iy = CartesianIndex(0,1)
    value = 1.
    out[I] = 0.0
    if I in Ids[begin]+Id:Ids[end]-Id
        out[I] = value * max(abs(G(2 * I - Ix, Ids) -  G(2 * I+Ix, Ids)) , abs(G(2 * I - Iy, Ids) -  G(2 * I+Iy, Ids)))
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


@kernel function left2(out)
    I = @index(Global, Cartesian)
    Ids = CartesianIndices(out)
    Id = oneunit(I)
    Ix = CartesianIndex(1,0)
    Iy = CartesianIndex(0,1)
    value = 1.
    out[I] = 0.0
    if I in Ids[begin]+Id:Ids[end]-Id
        out[I] = value * abs(G(2 * I - 2*Ix, Ids) -  G(2 * I - Ix, Ids))
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
