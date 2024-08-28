using Plots
using oneAPI
using KernelAbstractions
using KernelAbstractions.Extras.LoopInfo: @unroll
using Base: Callable
include("derrivative-kernels.jl")
include("elyps.jl")
include("gauss-seidel.jl")
include("explicit.jl")
include("boundary-conditions.jl")

Arrtype = oneArray
arr = Arrtype(zeros(Float32,256, 256))
SIZE = 254
M = testdata(SIZE, 6, SIZE / 5, 2)

Inds = CartesianIndices(arr)
Id = one(Inds[begin])

arr[Inds[begin]+Id:Inds[end]-Id] = Arrtype(M)

function solve(initialCondition::T, timesteps::Int ; arrtype=T) where T<:AbstractArray
    # variable<s
    α::Float32 = 2.e6
    h::Float32 = 3f-3 * 64 / size(initialCondition)[1]
    Δt::Float32 = 1e-3
    ε::Float32 = 1e-3
    W′(x) = -x * (1 - x^2)
    device = get_backend(initialCondition)

    C = rand(Float32, size(initialCondition)...) |> arrtype
    M = zeros(Float32, size(initialCondition)...) |> arrtype
    Ξ = zeros(Float32, size(initialCondition)...) |> arrtype
    Ψ = zeros(Float32, size(initialCondition)...) |> arrtype
    tmp = zeros(Float32, size(initialCondition)...) |> arrtype

    Dirac = zeros(Float32, size(initialCondition)...) |> arrtype
    Neumann1 = zeros(Float32, size(initialCondition)...) |> arrtype
    Neumann2 = zeros(Float32, size(initialCondition)...) |> arrtype


    l = BoundaryKernels.left(device , 128 , size(initialCondition))
    r = BoundaryKernels.right(device , 128 , size(initialCondition))
    t = BoundaryKernels.top(device , 128 , size(initialCondition))
    b = BoundaryKernels.bottom(device , 128 , size(initialCondition))


    Φ = copy(initialCondition) |> arrtype
    stencil = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    stencil = CartesianIndex.(stencil)
    stencil = stencil |> arrtype
    ellipical_solver = elyps_solver!(device, 256, size(C))
    jacoby_step = relaxed_jacoby!(device, 256, size(C))

    l(tmp)
    Neumann2 += tmp
    r(tmp)
    Neumann2 += tmp
    Neumann2 *= 1f-0 * h^2
    #print(Neumann1)

    @showprogress for j = 1:timesteps
        set_xi_and_psi!(Ξ, Ψ, Φ, W′, Δt)
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


function animated_solve(initialCondition::T, timesteps::Int, filepath::String ; arrtype=T) where T<:AbstractArray
    # variable<s
    α::Float32 = 2.e6
    h::Float32 = 3f-3 * 64 / size(initialCondition)[1]
    Δt::Float32 = 1e-2
    ε::Float32 = 1e-3
    W′(x) = -x * (1 - x^2)
    device = get_backend(initialCondition)

    C = rand(Float32, size(initialCondition)...) |> arrtype
    M = zeros(Float32, size(initialCondition)...) |> arrtype
    Ξ = zeros(Float32, size(initialCondition)...) |> arrtype
    Ψ = zeros(Float32, size(initialCondition)...) |> arrtype
    tmp = zeros(Float32, size(initialCondition)...) |> arrtype

    Dirac = zeros(Float32, size(initialCondition)...) |> arrtype
    Neumann1 = zeros(Float32, size(initialCondition)...) |> arrtype
    Neumann2 = zeros(Float32, size(initialCondition)...) |> arrtype


    l = BoundaryKernels.left(device , 128 , size(initialCondition))
    r = BoundaryKernels.right(device , 128 , size(initialCondition))
    t = BoundaryKernels.top(device , 128 , size(initialCondition))
    b = BoundaryKernels.bottom(device , 128 , size(initialCondition))


    Φ = initialCondition |> arrtype
    stencil = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    stencil = CartesianIndex.(stencil)
    stencil = stencil |> arrtype
    ellipical_solver = elyps_solver!(device, 256, size(C))
    jacoby_step = relaxed_jacoby!(device, 256, size(C))

    l(tmp)
    Neumann2 += tmp
    r(tmp)
    Neumann2 +=tmp
    Neumann2 *= 1f-0 * h^2
    #print(Neumann1)

    anim=@animate for j = 1:timesteps
        heatmap(Array(Φ))
        set_xi_and_psi!(Ξ, Ψ, Φ, W′, Δt)
        # add boundary conditions
        Ψ .+= add_boundary(Φ , h , Dirac , Neumann1 , Neumann2)
        for _ = 1:100
            ellipical_solver(C, Φ, α, h, stencil, 10)
            KernelAbstractions.synchronize(device)
            jacoby_step(Φ, M, Ξ, Ψ, C, h, α, ε, Δt, 10)
            KernelAbstractions.synchronize(device)
        end
    end
    mp4(anim , filepath , fps=24)
    return nothing

end
