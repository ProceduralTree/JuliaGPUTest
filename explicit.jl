using Base: Callable
using CUDA
using KernelAbstractions

include("derrivative-kernels.jl")
include("boundary-conditions.jl")

function set_xi_and_psi!(
    Ξ::CuArray,
    Ψ::CuArray,
    Φ::CuArray,
    W′::Callable,
    Δt::Float32)

    xi_init(x) = x / Δt
    psi_init(x) = W′(x) - 2 * x
    Ξ[2:end-1, 2:end-1] = xi_init.(Φ[2:end-1, 2:end-1])
    Ψ[2:end-1, 2:end-1] = psi_init.(Φ[2:end-1, 2:end-1])
    return nothing
end

function add_boundary(Φ,h::Float32, Dirac::CuArray, Neumann1::CuArray, Neumann2::CuArray)
    device = get_backend(Φ)
    out = KernelAbstractions.zeros(device,Float32,size(Φ)...)
    tmp = KernelAbstractions.zeros(device,Float32,size(Φ)...)

    add_divergence = divergence_add_kernel(device, 128, size(out))
    add_gradient = gradient_add_kernel(device, 128, size(out))

    Dirac_kernel = Dirac(device , 128 , size(out))
    Neumann1_kernel = Neumann1(device , 128 , size(out))
    Neumann2_kernel = Neumann2(device , 128 , size(out))

    Inds = CartesianIndices(out)
    # add boundary conditions
    #
    Neumann2_kernel(tmp , Φ)
    out += tmp
    Neumann1_kernel(tmp , Φ)
    add_divergence(tmp, out,h)
    Dirac_kernel(tmp , Φ)
    add_gradient(tmp, out,h)
    return out
end
