
using KernelAbstractions
using KernelAbstractions.Extras.LoopInfo: @unroll
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

@kernel function relaxed_jacoby!(
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
    if I in (Ids[begin]+Id:Ids[end]-Id)
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
