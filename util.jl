using Base: Callable
using CUDA
using Random
function testdata(gridsize , blobs , radius ,norm;rng=MersenneTwister(42))
rngpoints = rand(rng,1:gridsize, 2, blobs)
M = zeros(gridsize,gridsize) .- 1
for p in axes(rngpoints , 2)
    point = rngpoints[:, p]
    for I in CartesianIndices(M)
        if (LinearAlgebra.norm(point .- I.I  , norm) < radius)
            M[I] = 1
        end
    end
end
M
end

# function bulk_energy(solver::T) where T <: Union{multi_solver , relaxed_multi_solver}
#     energy = 0
#     dx = CartesianIndex(1,0)
#     dy = CartesianIndex(0,1)
#     W(x) = 1/4 * (1-x^2)^2
#     for I in CartesianIndices(solver.phase)[2:end-1,2:end-1]
#         i,j = I.I
#         energy += solver.epsilon^2 / 2 * G(i+ 0.5,j ,solver.len, solver.width) * (solver.phase[I+dx] - solver.phase[I])^2 + G(i,j+0.5,solver.len ,solver.width) * (solver.phase[I+dy] - solver.phase[I])^2 + W(solver.phase[I])
#         end
#    return energy
# end

function massbal(arr)
    num_cells= *((size(arr).-2)...)
    return sum(arr[2:end-1, 2:end-1])/num_cells
    end


