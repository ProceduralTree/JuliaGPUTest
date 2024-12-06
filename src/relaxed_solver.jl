using ProgressMeter
using CUDA
using KernelAbstractions
using Base: Callable
include("derrivative-kernels.jl")
include("gauss-seidel.jl")
include("explicit.jl")
include("boundary-conditions.jl")
include("initial_conditions.jl")
include("elyps.jl")

function solve(initialCondition::T, timesteps::Int ; arrtype=T) where T<:AbstractArray
    device = get_backend(initialCondition)

    α = 1f7

    C = rand(Float32, size(initialCondition)...) |> arrtype
    M = zeros(Float32, size(initialCondition)...) |> arrtype
    Ξ = zeros(Float32, size(initialCondition)...) |> arrtype
    Ψ = zeros(Float32, size(initialCondition)...) |> arrtype
    tmp = zeros(Float32, size(initialCondition)...) |> arrtype

    DynamicBD = zeros(Float32, size(initialCondition)...) |> arrtype
    Neumann2 = zeros(Float32, size(initialCondition)...) |> arrtype


    b = border(device, 256 , size(initialCondition))


    Φ = copy(initialCondition) |> arrtype
    stencil = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    stencil = CartesianIndex.(stencil)
    stencil = stencil |> cu
    ellipical_solver = elyps_solver!(device, 256, size(C))
    jacoby_step = relaxed_jacoby!(device, 256, size(C))

    b(tmp)
    Neumann2 += -7.5f-1 * tmp

    @showprogress for j = 1:timesteps
        set_xi_and_psi!(Ξ, Ψ, Φ, W′, Δt)
        DynamicBD = Neumann2 .* Φ
        # add boundary conditions
        Ψ .+= Neumann2
        for _ = 1:100
            ellipical_solver(C, Φ, α, h, stencil, 10)
            KernelAbstractions.synchronize(device)
            jacoby_step(Φ, M, Ξ, Ψ, C, h, α, ε, Δt, 10)
            KernelAbstractions.synchronize(device)
        end
    end
    return Φ


end
