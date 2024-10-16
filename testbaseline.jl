using Plots
using ProgressMeter
#using oneAPI
using CUDA
using KernelAbstractions
using KernelAbstractions.Extras.LoopInfo: @unroll
using Base: Callable
include("derrivative-kernels.jl")
include("elyps.jl")
include("gauss-seidel.jl")
include("explicit.jl")
include("boundary-conditions.jl")

Arrtype = cu
arr = Arrtype(zeros(Float32,256, 256))
SIZE = 254
M = testdata(SIZE, 6, SIZE / 5, 2)

Inds = CartesianIndices(arr)
Id = one(Inds[begin])

arr[Inds[begin]+Id:Inds[end]-Id] = Arrtype(M)

function solve(initialCondition::T, timesteps::Int ; arrtype=T) where T<:AbstractArray
    # variable<s
    h::Float32 = 3f-3 * 64 / size(initialCondition)[1]
    Δt::Float32 = 1e-3
    ε::Float32 = 1e-3
    W′(x) = -x * (1 - x^2)
    device = get_backend(initialCondition)

    M = zeros(Float32, size(initialCondition)...) |> arrtype
    Ξ = zeros(Float32, size(initialCondition)...) |> arrtype
    Ψ = zeros(Float32, size(initialCondition)...) |> arrtype
    tmp = zeros(Float32, size(initialCondition)...) |> arrtype

    DynamicBD = zeros(Float32, size(initialCondition)...) |> arrtype
    Neumann2 = zeros(Float32, size(initialCondition)...) |> arrtype


    l = BoundaryKernels.left(device , 128 , size(initialCondition))
    r = BoundaryKernels.right(device , 128 , size(initialCondition))
    t = BoundaryKernels.top(device , 128 , size(initialCondition))
    b = BoundaryKernels.bottom(device , 128 , size(initialCondition))


    Φ = copy(initialCondition) |> arrtype
    println(sum(Φ))
    jacoby_step = jacoby!(device, 256, size(Φ))

    l(tmp)
    Neumann2 += -7.5f-1 * tmp

    @showprogress for j = 1:timesteps
        set_xi_and_psi!(Ξ, Ψ, Φ, W′, Δt)
        DynamicBD = Neumann2
        # add boundary conditions
        #Ψ .+= DynamicBD
        jacoby_step(Φ, M, Ξ, Ψ, h, ε, Δt, 10000)
        KernelAbstractions.synchronize(device)
    end
    println(sum(Φ))
    return Φ

end


function animated_solve(initialCondition::T, timesteps::Int, filepath::String ; arrtype=T) where T<:AbstractArray
    # variable<s
    h::Float32 = 3f-4
    Δt::Float32 = 1e-4
    ε::Float32 = 2e-4
    W′(x) = -x * (1 - x^2)

    M = zeros(Float32, size(initialCondition)...) |> arrtype
    Ξ = zeros(Float32, size(initialCondition)...) |> arrtype
    Ψ = zeros(Float32, size(initialCondition)...) |> arrtype
    tmp = zeros(Float32, size(initialCondition)...) |> arrtype

    Neumann2 = zeros(Float32, size(initialCondition)...) |> arrtype
    DynamicBD = zeros(Float32, size(initialCondition)...) |> arrtype
    device = get_backend(initialCondition)


    l = BoundaryKernels.left(device , 128 , size(initialCondition))
    l2 = BoundaryKernels.left2(device , 128 , size(initialCondition))
    r = BoundaryKernels.right(device , 128 , size(initialCondition))
    t = BoundaryKernels.top(device , 128 , size(initialCondition))
    b = BoundaryKernels.bottom(device , 128 , size(initialCondition))


    Φ = initialCondition |> arrtype
    jacoby_step = jacoby!(device, 256, size(Φ))

    mass = []
    Neumann2 += -7.5f-1 * tmp
    p = Progress(timesteps)
    anim=@animate for j = 1:timesteps
        #heatmap(Array(Φ), aspect_ratio=:equal , clims=(-1,1))
        push!(mass , sum(Φ))
        plot(mass)
        set_xi_and_psi!(Ξ, Ψ, Φ, W′, Δt)
        DynamicBD = Neumann2
        # add boundary conditions
        #Ψ .+= DynamicBD
        jacoby_step(Φ, M, Ξ, Ψ, h, ε, Δt, 10000)
        KernelAbstractions.synchronize(device)
        next!(p)
    end
    mp4(anim , filepath , fps=24)
    return nothing

end
