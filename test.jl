using Plots
using CUDA
using KernelAbstractions
using KernelAbstractions.Extras.LoopInfo: @unroll
using Base: Callable
include("derrivative-kernels.jl")
include("elyps.jl")
include("gauss-seidel.jl")
include("explicit.jl")
include("boundary-conditions.jl")
arr = cu(zeros(1024, 1024))
SIZE = 1022
M = testdata(SIZE, 16, SIZE / 5, 2)

Inds = CartesianIndices(arr)
Id = one(Inds[begin])

arr[Inds[begin]+Id:Inds[end]-Id] = cu(M)

function solve(initialCondition::CuArray, timesteps::Int)
    # variables
    α::Float32 = 2.e6
    h::Float32 = 3f-3 * 64 / size(initialCondition)[1]
    Δt::Float32 = 1e-3
    ε::Float32 = 1e-3
    W′(x) = -x * (1 - x^2)
    device = get_backend(initialCondition)

    device_arr = cu
    C = rand(Float32, size(initialCondition)...) |> device_arr
    M = zeros(Float32, size(initialCondition)...) |> device_arr
    Ξ = zeros(Float32, size(initialCondition)...) |> device_arr
    Ψ = zeros(Float32, size(initialCondition)...) |> device_arr

    Dirac = zeros(Float32, size(initialCondition)...) |> device_arr
    Neumann1 = zeros(Float32, size(initialCondition)...) |> device_arr
    Neumann2 = zeros(Float32, size(initialCondition)...) |> device_arr


    l = BoundaryKernels.left(device , 128 , size(out))
    r = BoundaryKernels.right(device , 128 , size(out))
    f = BoundaryKernels.flow(device , 128 , size(out))


    Φ = initialCondition |> device_arr
    stencil = [(1, 0), (-1, 0), (0, 1), (0, -1)] |> device_arr
    stencil = CartesianIndex.(stencil)
    ellipical_solver = elyps_solver!(device, 256, size(C))
    jacoby_step = relaxed_jacoby!(device, 256, size(C))


    Neumann1 = 1f-3 .* (l(Dirac) + r(Dirac))



    @showprogress for j = 1:timesteps
        set_xi_and_psi!(Ξ, Ψ, Φ, W′, Δt,h)
        # add boundary conditions
        Ψ .+= add_boundary(Φ , h , Dirac , Neumann1 , Neumann2)
        for _ = 1:100
            ellipical_solver(C, Φ, α, h, stencil, 10)
            KernelAbstractions.synchronize(device)
            jacoby_step(Φ, M, Ξ, Ψ, C, h, α, ε, Δt, 10)
            KernelAbstractions.synchronize(device)
        end
    end
    return Φ

end

function animated_solve(initialCondition::CuArray, timesteps::Int, filepath::String)
    # variables
    α::Float32 = 2.e6
    h::Float32 = 3e-3 * 64 / 1024
    Δt::Float32 = 1e-3
    ε::Float32 = 1e-3
    W′(x) = -x * (1 - x^2)

    device_arr = cu
    C = rand(Float32, size(initialCondition)...) |> device_arr
    M = zeros(Float32, size(initialCondition)...) |> device_arr
    Ξ = zeros(Float32, size(initialCondition)...) |> device_arr
    Ψ = zeros(Float32, size(initialCondition)...) |> device_arr
    Φ = initialCondition |> device_arr
    o = zeros(Float32, size(C)...) |> device_arr
    stencil = [(1, 0), (-1, 0), (0, 1), (0, -1)] |> device_arr
    stencil = CartesianIndex.(stencil)
    device = get_backend(C)
    ellipical_solver = elyps_solver!(device, 256, size(C))
    jacoby_step = relaxed_jacoby!(device, 256, size(C))

    anim = @animate for j = 1:timesteps
            heatmap(Array(Φ), aspect_ratio=:equal)
        set_xi_and_psi!(Ξ, Ψ, Φ, W′, Δt, z, z, z)
        for _ = 1:100
            ellipical_solver(C, Φ, α, h, stencil, 10)
            KernelAbstractions.synchronize(device)
            jacoby_step(Φ, M, Ξ, Ψ, C, h, α, ε, Δt, 10)
            KernelAbstractions.synchronize(device)
        end
    end
    mp4(anim, filepath, fps=24)
end


function mass_solve(initialCondition::CuArray, timesteps::Int, filepath::String)
    # variables
    α::Float32 = 2.e6
    h::Float32 = 3e-3 * 64 / 1024
    Δt::Float32 = 1e-3
    ε::Float32 = 1e-3
    W′(x) = -x * (1 - x^2)

    device_arr = cu
    C = rand(Float32, size(initialCondition)...) |> device_arr
    M = zeros(Float32, size(initialCondition)...) |> device_arr
    Ξ = zeros(Float32, size(initialCondition)...) |> device_arr
    Ψ = zeros(Float32, size(initialCondition)...) |> device_arr
    Φ = initialCondition |> device_arr
    o = zeros(Float32, size(C)...) |> device_arr
    stencil = [(1, 0), (-1, 0), (0, 1), (0, -1)] |> device_arr
    stencil = CartesianIndex.(stencil)
    device = get_backend(C)
    ellipical_solver = elyps_solver!(device, 256, size(C))
    jacoby_step = relaxed_jacoby!(device, 256, size(C))
    mbal = []
    @showprogress for j = 1:timesteps
        push!(mbal , massbal(Φ))
        set_xi_and_psi!(Ξ, Ψ, Φ, W′, Δt, z, z, z)
        for _ = 1:100
            ellipical_solver(C, Φ, α, h, stencil, 10)
            KernelAbstractions.synchronize(device)
            jacoby_step(Φ, M, Ξ, Ψ, C, h, α, ε, Δt, 10)
            KernelAbstractions.synchronize(device)
        end
    end
    plot(mbal)
end
