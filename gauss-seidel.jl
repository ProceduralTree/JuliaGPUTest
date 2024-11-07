
using KernelAbstractions
using KernelAbstractions.Extras.LoopInfo: @unroll
using Random
#using oneAPI: oneArray
using LinearAlgebra
using ProgressMeter

include("util.jl")


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
    if I in ((Ids[begin]+Id):(Ids[end]-Id))
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

@kernel function jacoby!(
    Φ,
    M,
    @Const(Ξ),
    @Const(Ψ),
    @Const(h),
    @Const(ε),
    @Const(Δt),
    @Const(iterations)
)
    I   = @index(Global, Cartesian)
    Id  = oneunit(I)
    Ids = CartesianIndices(M)
    Ix = CartesianIndex(1, 0)
    Iy = CartesianIndex(0, 1)
    if I in (Ids[begin]+Id:Ids[end]-Id)
        g = G(2 * I + Ix, Ids) + G(2 * I + Iy, Ids) + G(2 * I - Ix, Ids) + G(2 * I - Iy, Ids)
        a1 = 1/Δt
        a2 = -1* ε^2/h^2 * g  - 2
        b1 = 1/h^2 * g
        b2 = 1
        for _ = 1:iterations

            Σμ = G(2 * I + Ix, Ids) * M[I+Ix] + G(2 * I + Iy, Ids) * M[I+Iy] + G(2 * I - Ix, Ids) * M[I-Ix] + G(2 * I - Iy, Ids) * M[I-Iy]

            Σϕ = G(2 * I + Ix, Ids) * Φ[I+Ix] + G(2 * I + Iy, Ids) * Φ[I+Iy] +G(2 * I - Ix, Ids) * Φ[I-Ix] +G(2 * I - Iy, Ids) * Φ[I-Iy]

            c1 = Ξ[I] + 1/h^2   * Σμ
            c2 = Ψ[I] - ε^2/h^2 * Σϕ

            # stupid matrix solve
            @inline Φ[I] =  (c1*b2 - c2*b1) / (a1*b2 - a2*b1)
            @inline M[I] = (a1*c2 - a2*c1) / (a1*b2 - a2*b1)
            #
            @synchronize()
        end

    end
end
