using KernelAbstractions

@kernel function border_halo(out, radius::Int)
    I = @index(Global, Cartesian)
    Ids = CartesianIndices(out)
    Id = oneunit(I)
    Ix = CartesianIndex(1,0)
    Iy = CartesianIndex(0,1)
    value = 1.
    out[I] = 0.0
    if I in Ids[begin]+Id:Ids[end]-Id
        out[I] = value * max(
            abs(G(2 * I - radius*Ix, Ids) -  G(2 * I+ radius * Ix, Ids)) ,
            abs(G(2 * I - radius * Iy, Ids) -  G(2 * I+ radius * Iy, Ids)))
    end

end




@kernel function calculate_angle(out, @Const(Φ),@Const(f))
    I = @index(Global , Cartesian)
    Ids = CartesianIndices(out)
    Id = oneunit(I)
    Ix = CartesianIndex(1,0)
    Iy = CartesianIndex(0,1)
    # near boundary?
    if I in Ids[begin]+Id:Ids[end]-Id
        @inline dx = G(2 * I + Ix, Ids) * (Φ[I+Ix] - Φ[I])
        @inline dy = G(2 * I + Iy, Ids) * (Φ[I+Iy] - Φ[I])
        @inline out[I] =(dx^2 + dy^2) * f[I]
    end

end
