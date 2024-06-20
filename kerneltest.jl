using KernelAbstractions
using oneAPI: oneArray

device_arr = x -> x

function G(I::CartesianIndex , Inds::CartesianIndices)
    if I in Inds
        return 1
       end
    return 0
    end

@kernel function test(@Const(A) , output)
    I = @index(Global , Cartesian)
    Id = oneunit(I)
    output[I] = 0
    Idx = CartesianIndex(1,0)
    Idy = CartesianIndex(0,1)
    Ids = CartesianIndices(A)
    output[I] = 0
    if I in Ids[begin]+Id:Ids[end]-Id
         @inline output[I] += A[I + Idy] + A[I + Idx]
         @inline output[I] += A[I - Idy] + A[I - Idx]
         @inline output[I] -= 4 * A[I]
    end
    end

indicies = [Idx , Idy , - Idx , -Idy]
A = rand(Float32 , (10_000,10_000)) |> device_arr
o = zeros(Float32 , size(A))|> device_arr
device = get_backend(A)
test_kernel = test(device , 256 ,size(A))
test_kernel(A,o)
synchronize(device)
o
