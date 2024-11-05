using Plots
using ProgressMeter
#using oneAPI
using oneAPI
using KernelAbstractions
using KernelAbstractions.Extras.LoopInfo: @unroll
using Base: Callable
include("derrivative-kernels.jl")
include("elyps.jl")
include("gauss-seidel.jl")
include("explicit.jl")
include("boundary-conditions.jl")



arr = Arrtype(zeros(Float32,265, 512))
dev = get_backend(arr)
SIZE = 254
#M = testdata(SIZE, 6, SIZE / 5, 2)
M = KernelAbstractions.zeros(dev , Float32 , size(arr) .- 2)
fill!(M , -1)
circle = set_circle(dev , 128  , size(M))
circle(M , 2f2 , CartesianIndex(1, 400))
circle(M , 1f2 , CartesianIndex(1, 800))
Inds = CartesianIndices(arr)
Id = one(Inds[begin])

arr[Inds[begin]+Id:Inds[end]-Id] = M


Arrtype = oneArray
arr = Arrtype(zeros(Float32,256, 256))
SIZE = 254
M = testdata(SIZE, 4, SIZE / 5, 2)

Inds = CartesianIndices(arr)
Id = one(Inds[begin])

arr[Inds[begin]+Id:Inds[end]-Id] = Arrtype(M)
d = BoundaryKernels.domain(get_backend(arr) , 128 , size(arr))
d(arr)
function solve(initialCondition::T, timesteps::Int ; arrtype=T , θ=0) where T<:AbstractArray
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
    b = BoundaryKernels.border(device , 128 , size(initialCondition))


    Φ = copy(initialCondition) |> arrtype
    println(sum(Φ))
    jacoby_step = jacoby!(device, 256, size(Φ))

    b(tmp)
    Neumann2 += θ * tmp

    @showprogress for j = 1:timesteps
        set_xi_and_psi!(Ξ, Ψ, Φ, W′, Δt)
        DynamicBD = Neumann2 * (1 .- Φ.^2)
        # add boundary conditions
        Ψ .+= DynamicBD
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


    b = BoundaryKernels.border(device , 128 , size(initialCondition))


    Φ = initialCondition |> arrtype
    jacoby_step = jacoby!(device, 256, size(Φ))

    mass = []
    b(tmp)
    Neumann2 += -5f-1 * tmp
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
