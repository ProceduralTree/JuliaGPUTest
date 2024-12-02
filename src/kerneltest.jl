using BenchmarkTools
using CUDA
using KernelAbstractions
using KernelAbstractions.Extras.LoopInfo: @unroll
using LinearAlgebra
using Random

device_arr = CuArray
# device_arr = x -> x
function G(I::CartesianIndex, Inds::CartesianIndices)
    Id = oneunit(I)
    if I in 2*(Inds[begin]+Id):2*(Inds[end]-Id)
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


@kernel function glap(@Const(A), output, stencil)
    I = @index(Global, Cartesian)
    Id = oneunit(I)
    output[I] = 0
    Idx = CartesianIndex(1, 0)
    Idy = CartesianIndex(0, 1)
    Ids = CartesianIndices(A)
    output[I] = 0
    if I in Ids[begin]+Id:Ids[end]-Id
        @unroll for i = stencil
            @inline output[I] += G(2 * I + i, Ids) * (A[I+i] - A[I])
        end
    end
end

@kernel function elyps_solver(@Const(C), @Const(PHI), output, alpha::Float32, h::Float32, @Const(stencil))
    I = @index(Global, Cartesian)
    Id = oneunit(I)
    Ids = CartesianIndices(C)
    output[I] = 0
    g = 0
    c = 0.0
    if I in Ids[begin]+Id:Ids[end]-Id
        @unroll for i = stencil
            g += G(2 * I + i, Ids)
            @inline c += G(2 * I + i, Ids) * C[I+i]
        end
    end
    dv = g + alpha * h^2
    @inline output[I] = (alpha * h^2 * PHI[I] + c) / dv

end
function nokernel_elyps_solver!(PHI, n)
    Inds = CartesianIndices(PHI)
    Id = oneunit(Inds[begin])
    C = rand(Float32 ,size(PHI))
    alpha::Float32 = 2.e6
    h::Float32 = 3e-3 * 64 / 1024
    stencil = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    stencil = CartesianIndex.(stencil)
    for k in 1:n
        for I in Inds[begin]+Id:Inds[end]-Id

            g = 0.
            c = 0.
            for i = stencil
                g += G(2 * I + i, Inds)
                @inline c += G(2 * I + i, Inds) * C[I+i]
            end
            dv = g + alpha * h^2
            @inline C[I] = (alpha * h^2 * PHI[I] + c) / dv
        end

    end
    C
end


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

function test_glap(n; device_arr=(x) -> x)
    A = rand(Float32, (1024, 1024)) |> device_arr
    G = rand(Float32, 2 .* (1024, 1024)) |> device_arr
    o = zeros(Float32, size(A)) |> device_arr
    stencil = [(1, 0), (-1, 0), (0, 1), (0, -1)] |> device_arr
    stencil = CartesianIndex.(stencil)
    device = get_backend(A)
    test_kernel = glap(device, 64, size(A))
    total = 0.0
    for i in 1:n
        test_kernel(A, o, stencil)
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

function test_elyps_solver(arr, n; device_arr=(x) -> x)
    C = rand(Float32, (1024, 1024)) |> device_arr
    PHI = arr |> device_arr
    o = zeros(Float32, size(C)) |> device_arr
    stencil = [(1, 0), (-1, 0), (0, 1), (0, -1)] |> device_arr
    stencil = CartesianIndex.(stencil)
    device = get_backend(C)
    test_kernel = elyps_solver(device, 256, size(C))
    alpha::Float32 = 2.e6
    h::Float32 = 3e-3 * 64 / 1024
    for i in 1:n
        test_kernel(C, PHI, o, alpha, h, stencil)
        KernelAbstractions.synchronize(device)
        C, o = o, C
    end
    return C
end

function testdata(gridsize, blobs, radius, norm; rng=MersenneTwister(42))
    rngpoints = rand(rng, 1:gridsize, 2, blobs)
    M = zeros(gridsize, gridsize) .- 1
    for p in axes(rngpoints, 2)
        point = rngpoints[:, p]
        for I in CartesianIndices(M)
            if (LinearAlgebra.norm(point .- I.I, norm) < radius)
                M[I] = 1
            end
        end
    end
    M
end


