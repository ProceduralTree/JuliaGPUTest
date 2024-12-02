
using CUDA
using KernelAbstractions

@kernel function elyps_solver(@Const(C), @Const(Φ), output, α::Float32, h::Float32, @Const(stencil))
    I = @index(Global, Cartesian)
    Id = oneunit(I)
    Ids = CartesianIndices(C)
    output[I] = 0
    g = 0
    c = 0.0
    if I in Ids[begin]+Id:Ids[end]-Id
        @unroll for i = stencil
            g += G(2 * I + i, Ids)
            @inline c += G(2 * I + i, Ids) * C[I+i]
        end
    end
    dv = g + α * h^2
    @inline output[I] = (α * h^2 * Φ[I] + c) / dv

end
