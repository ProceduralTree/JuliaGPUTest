
using KernelAbstractions
@kernel function elyps_solver!(C, @Const(PHI), alpha::Float32, h::Float32, @Const(stencil), iterations::Int)
    I = @index(Global, Cartesian)
    Id = oneunit(I)
    Ids = CartesianIndices(C)
    g = 0
    c = 0.0
    if I in Ids[begin]+Id:Ids[end]-Id
        for _ in iterations
            @unroll for i = stencil
                g += G(2 * I + i, Ids)
                @inline c += G(2 * I + i, Ids) * C[I+i]
            end
            dv = g + alpha * h^2
            @inline C[I] = (alpha * h^2 * PHI[I] + c) / dv
        end
    end
end
