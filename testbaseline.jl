using CUDA
using Revise
includet("domains.jl")
using .Domains:circle as G
include("initial_conditions.jl")
arr = _init(SIZE=(1024,1024))
include("solvers.jl")
solve(arr ,1000)
