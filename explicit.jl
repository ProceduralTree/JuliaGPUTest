using Base: Callable
using KernelAbstractions

include("derrivative-kernels.jl")
include("boundary-conditions.jl")

function set_xi_and_psi!(
    Ξ::T,
    Ψ::T,
    Φ::T,
    W′::Callable,
    Δt::Float32) where T<:AbstractArray

    xi_init(x) = x / Δt
    psi_init(x) = W′(x) - 2 * x
    Ξ[2:end-1, 2:end-1] = xi_init.(Φ[2:end-1, 2:end-1])
    Ψ[2:end-1, 2:end-1] = psi_init.(Φ[2:end-1, 2:end-1])
    return nothing
end

function add_boundary(Φ,h::Float32, Dirac::T, Neumann1::T, Neumann2::T) where T<:AbstractArray
    device = get_backend(Φ)
    out = KernelAbstractions.zeros(device,Float32,size(Φ)...)

    add_divergence = divergence_add_kernel(device, 128, size(out))
    add_gradient = gradient_add_kernel(device, 128, size(out))
    #
    out += Neumann2
    add_divergence(Neumann1, out,h)
    add_gradient(Dirac, out,h)
    return out
end
