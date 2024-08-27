using KernelAbstractions
function G(I::CartesianIndex, Inds::CartesianIndices)
    Id = oneunit(I)
    if I in 2*(Inds[begin]+Id):2*(Inds[end]-Id)
        return 1
    end
    return 0
end

@kernel function gdiv(@Const(A), output)
    I = @index(Global, Cartesian)
    Id = oneunit(I)
    output[I] = 0
    Idx = CartesianIndex(1, 0)
    Idy = CartesianIndex(0, 1)
    Ids = CartesianIndices(A)
    output[I] = 0
    if I in Ids[begin]+Id:Ids[end]-Id
        @inline output[I] += G(2 * I + i, Ids) * (A[I+Idx] - A[I]) + G(2 * I + i, Ids) * (A[I+Idy] - A[I])
    end
end

@kernel function divergence_add_kernel(@Const(A), output,h::Float32)
    I = @index(Global, Cartesian)
    Id = oneunit(I)
    Idx = CartesianIndex(1, 0)
    Idy = CartesianIndex(0, 1)
    Ids = CartesianIndices(A)
    if I in Ids[begin]+Id:Ids[end]-Id
        @inline output[I] += 0.5*(A[I+Idx] - A[I-Idx] + A[I+Idy] - A[I-Idy])
        @inline output[I] *= 1/h
    end
end

@kernel function is_boundary(output)
    I = @index(Global, Cartesian)
    Id = oneunit(I)
    output[I] = 0
    Idx = CartesianIndex(1, 0)
    Idy = CartesianIndex(0, 1)
    Ids = CartesianIndices(output)
    output[I] = 0
    if I in Ids[begin]+Id:Ids[end]-Id

        @inline output[I] += G(2 * I + Idx, Ids) +  G(2 * I + Idy, Ids) + G(2 * I - Idx, Ids) +  G(2 * I - Idy, Ids) - 4*G(2 * I, Ids)

    end
end

@kernel function gradient_add_kernel(@Const(A), output,h::Float32)
    I = @index(Global, Cartesian)
    Id = oneunit(I)
    Idx = CartesianIndex(1, 0)
    Idy = CartesianIndex(0, 1)
    Ids = CartesianIndices(A)
    if I in Ids[begin]+Id:Ids[end]-Id
        @inline output[I] += A[I+Idx] + A[I+Idy] + A[I-Idx] + A[I-Idy] - 4 * A[I]
        @inline output[I] *= 1/h^2
    end
end