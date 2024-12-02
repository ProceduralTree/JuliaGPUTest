using KernelAbstractions
using KernelAbstractions.Extras.LoopInfo: @unroll
using CUDA
using Random
#using oneAPI: oneArray
using LinearAlgebra
using ProgressMeter

include("util.jl")
include("elyps.jl")
include("gauss-seidel.jl")
include("derrivative-kernels.jl")

function G(I::CartesianIndex, Inds::CartesianIndices)
    Id = oneunit(I)
    if I in 2*(Inds[begin]+Id):2*(Inds[end]-Id)
        return 1
    end
    return 0
end

function gauss_seidel_CH(arr, n; device_arr=Array)
    # variables
    α::Float32 = 2.e6
    h::Float32 = 3e-3 * 64 / 1024
    Δt::Float32 = 1e-3
    ε::Float32 = 1e-3
    SIZE = (1024, 1024)
    W′(x) = -x * (1 - x^2)


    C = rand(Float32, size(arr)...) |> device_arr
    M = zeros(Float32,size(arr)...) |> device_arr
    Ξ = zeros(Float32,size(arr)...) |> device_arr
    Ψ = zeros(Float32,size(arr)...) |> device_arr
    PHI = arr |> device_arr
    o = zeros(Float32, size(C)...) |> device_arr
    stencil = [(1, 0), (-1, 0), (0, 1), (0, -1)] |> device_arr
    stencil = CartesianIndex.(stencil)
    device = get_backend(C)
    ellipical_solver = elyps_solver!(device, 256, size(C))
    gauss_seidel_step = relaxed_jacoby!(device, 256, size(C))

    @showprogress for j = 1:n
         set_xi_and_psi!(Ξ, Ψ, PHI, W′, Δt)

        for _ = 1:100
            ellipical_solver(C, PHI, α, h, stencil, 10)
            KernelAbstractions.synchronize(device)
            gauss_seidel_step(PHI, M, Ξ, Ψ, C, h, α, ε, Δt, 10)
            KernelAbstractions.synchronize(device)
        end
    end
    return PHI
end

arr = cu(zeros(1024, 1024))
SIZE = 1022
M = testdata(SIZE, 16, SIZE / 5, 2)

Inds = CartesianIndices(arr)
Id = one(Inds[begin])

arr[Inds[begin]+Id:Inds[end]-Id] = cu(M)

dev = get_backend(arr)
diver = divergence(dev, 256, size(arr))
bd = is_boundary(dev , 256 , size(arr))
