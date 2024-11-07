
using Plots
using ProgressMeter
using CUDA
#using oneAPI
using KernelAbstractions
using KernelAbstractions.Extras.LoopInfo: @unroll
using Base: Callable
include("derrivative-kernels.jl")
include("gauss-seidel.jl")
include("explicit.jl")
include("boundary-conditions.jl")
include("initial_conditions.jl")


function solve(initialCondition::T, timesteps::Int ; arrtype=T , θ=0) where T<:AbstractArray
    # variables
    h::Float32 = 3f-4
    Δt::Float32 = 1e-4
    ε::Float32 = 2e-4
    W′(x) = -x * (1 - x^2)
    device = get_backend(initialCondition)

    M = zeros(Float32, size(initialCondition)...) |> arrtype
    Ξ = zeros(Float32, size(initialCondition)...) |> arrtype
    Ψ = zeros(Float32, size(initialCondition)...) |> arrtype
    tmp = zeros(Float32, size(initialCondition)...) |> arrtype

    DynamicBD = zeros(Float32, size(initialCondition)...) |> arrtype
    Neumann2 = zeros(Float32, size(initialCondition)...) |> arrtype


    b = border(device , 128 , size(initialCondition))


    Φ = copy(initialCondition) |> arrtype
    println(sum(Φ))
    jacoby_step = jacoby!(device, 256, size(Φ))

    b(tmp)
    Neumann2 += θ * tmp

    @showprogress for j = 1:timesteps
        set_xi_and_psi!(Ξ, Ψ, Φ, W′, Δt)
        DynamicBD = Neumann2 #* (1 .- Φ.^2)
        # add boundary conditions
        Ψ += DynamicBD
        jacoby_step(Φ, M, Ξ, Ψ, h, ε, Δt, 1000)
        KernelAbstractions.synchronize(device)
    end
    println(sum(Φ))
    return Φ

end


function animated_solve(initialCondition::T, timesteps::Int, filepath::String ; arrtype=T) where T<:AbstractArray
    # variables
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


    b = border(device , 128 , size(initialCondition))


    Φ = initialCondition |> arrtype
    jacoby_step = jacoby!(device, 256, size(Φ))

    mass = []
    b(tmp)
    Neumann2 += 0f0 * tmp
    p = Progress(timesteps)
    anim=@animate for j = 1:timesteps
        heatmap(Array(Φ), aspect_ratio=:equal , clims=(-1,1))
        #push!(mass , sum(Φ))
        #plot(mass)
        set_xi_and_psi!(Ξ, Ψ, Φ, W′, Δt)
        DynamicBD = Neumann2 # .* (1 .- Φ.^2)
        Ψ += DynamicBD
        # add boundary conditions
        jacoby_step(Φ, M, Ξ,Ψ , h, ε, Δt, 1000)
        KernelAbstractions.synchronize(device)
        next!(p)
    end
    mp4(anim , filepath , fps=24)
    return nothing

end
