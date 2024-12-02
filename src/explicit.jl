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

