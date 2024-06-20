using KernelAbstractions
using CUDA
using oneAPI: oneArray
using LinearAlgebra

device_arr = CuArray
# device_arr = x -> x
function G(I::CartesianIndex, Inds::CartesianIndices)
    if I in Inds
        return 1
    end
    return 0
end

function nokernel(A, output, I::CartesianIndex)
    Id = oneunit(I)
    output[I] = 0
    Idx = CartesianIndex(1, 0)
    Idy = CartesianIndex(0, 1)
    Ids = CartesianIndices(A)
    output[I] = 0
    if I in Ids[begin]+Id:Ids[end]-Id
        @inline output[I] += A[I+Idy] + A[I+Idx] + A[I-Idy] + A[I-Idx] + 4 * A[I]
    end
end

@kernel function test(@Const(A), output)
    I = @index(Global, Cartesian)
    Id = oneunit(I)
    output[I] = 0
    Idx = CartesianIndex(1, 0)
    Idy = CartesianIndex(0, 1)
    Ids = CartesianIndices(A)
    output[I] = 0
    if I in Ids[begin]+Id:Ids[end]-Id
        @inline output[I] += A[I+Idy] + A[I+Idx] + A[I-Idy] + A[I-Idx] + 4 * A[I]
    end
end

indicies = [Idx, Idy, -Idx, -Idy]
device = get_backend(A)
test_kernel(A, o)
KernelAbstractions.synchronize(device)
o

function test_lap(n; device_arr=(x) -> x)
    A = rand(Float32, (1024, 1024)) |> device_arr
    o = zeros(Float32, size(A)) |> device_arr
    device = get_backend(A)
    test_kernel = test(device, 64, size(A))
    total = 0.0
    for i in 1:n
        test_kernel(A, o)
        KernelAbstractions.synchronize(device)
        A .+= o
        total += norm(o)
    end
    return total
end

function test_nokernel(n)
    A = rand(Float32, (1024, 1024))
    o = zeros(Float32, size(A))
    total = 0.0
    for i = 1:n
        for I in CartesianIndices(A)
            nokernel(A, o, I)
        end
        A .+= o
        total += norm(o)
    end
    return total
end
