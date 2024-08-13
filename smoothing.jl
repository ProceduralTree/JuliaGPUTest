using KernelAbstractions
using KernelAbstractions.Extras.LoopInfo: @unroll
using CUDA
using Random
#using oneAPI: oneArray
using LinearAlgebra
using ProgressMeter

include("util.jl")

function G(I::CartesianIndex, Inds::CartesianIndices)
    Id = oneunit(I)
    if I in 2*(Inds[begin]+Id):2*(Inds[end]-Id)
        return 1
    end
    return 0
end

@kernel function SMOOTH!(
    Φ,
    M,
    @Const(Ξ),
    @Const(Ψ),
    @Const(C),
    @Const(h),
    @Const(α),
    @Const(ε),
    @Const(Δt),
    @Const(iterations)
)
    I = @index(Global, Cartesian)
    Id = oneunit(I)
    Ids = CartesianIndices(C)
    Idx = CartesianIndex(1, 0)
    Idy = CartesianIndex(0, 1)
    if I in Ids[begin]+Id:Ids[end]-Id
        g = G(2 * I + Idx, Ids) + G(2 * I + Idy, Ids) + G(2 * I - Idx, Ids) + G(2 * I - Idy, Ids)
        for _ = 1:iterations
            Σμ = G(2 * I + Idx, Ids) * M[I+Idx]
            +G(2 * I + Idy, Ids) * M[I+Idy]
            +G(2 * I - Idx, Ids) * M[I-Idx]
            +G(2 * I - Idy, Ids) * M[I-Idy]
            @inline Φ[I] = (ε^2 * α * C[I] + (1 / g) * (h^2 * Ξ[I] + Σμ) - Ψ[I]) / (ε^2 * α + 2 + h^2 / (g * Δt))
            @synchronize()
            @inline M[I] = (Φ[I] / Δt - Ξ[I] + Σμ / h^2) * (h^2 / g)
            @synchronize()
        end

    end
end

@kernel function elyps_solver!(C, @Const(PHI), alpha::Float32, h::Float32, @Const(stencil), iterations::Int)
    I = @index(Global, Cartesian)
    Id = oneunit(I)
    Ids = CartesianIndices(C)
    g = 0
    c = 0.0
    if I in Ids[begin]+Id:Ids[end]-Id
        for _ in iterations
            @unroll for i = stencil
                g += G(2 * I + i, Ids)
                @inline c += G(2 * I + i, Ids) * C[I+i]
            end
            dv = g + alpha * h^2
            @inline C[I] = (alpha * h^2 * PHI[I] + c) / dv
        end
    end
end
function gauss_seidel_CH(arr, n; device_arr=(x) -> x)
    # variables
    α::Float32 = 2.e6
    h::Float32 = 3e-3 * 64 / 1024
    Δt::Float32 = 1e-3
    ε::Float32 = 1e-3
    SIZE = (1024, 1024)
    W′(x) = -x * (1 - x^2)


    C = rand(Float32, SIZE...) |> device_arr
    M = zeros(Float32, SIZE...) |> device_arr
    Ξ = zeros(Float32, SIZE...) |> device_arr
    Ψ = zeros(Float32, SIZE...) |> device_arr
    PHI = arr |> device_arr
    o = zeros(Float32, size(C)...) |> device_arr
    stencil = [(1, 0), (-1, 0), (0, 1), (0, -1)] |> device_arr
    stencil = CartesianIndex.(stencil)
    device = get_backend(C)
    ellipical_solver = elyps_solver!(device, 256, size(C))
    gauss_seidel_step = SMOOTH!(device, 256, size(C))

    @showprogress for _ = 1:n
        set_xi_and_psi!(Ξ, Ψ, PHI, W′, Δt)

        for _ = 1:100
            ellipical_solver(C, PHI, α, h, stencil, 100)
            KernelAbstractions.synchronize(device)
            gauss_seidel_step(PHI, M, Ξ, Ψ, C, h, α, ε, Δt, 100)
            KernelAbstractions.synchronize(device)
        end
    end
    return PHI
end

arr = zeros(1024, 1024)
SIZE = 1022
M = testdata(SIZE, 16, SIZE / 5, 2)

Inds = CartesianIndices(arr)
Id = one(Inds[begin])

arr[Inds[begin]+Id:Inds[end]-Id] = M
