using KernelAbstractions
include("boundary-conditions.jl")
include("util.jl")


Arrtype = cu


function random_init(;arrtype=cu , SIZE=(256,256) )
arr = arrtype(rand(SIZE...) .* 2 .- 1)
d = domain(get_backend(arr) , 128 , size(arr))
d(arr)
    return arr
    end


function _init(;arrtype=cu , SIZE=(256,256) )
arr = arrtype(testdata(SIZE[1], 4, SIZE[1] / 5, 2))
d = domain(get_backend(arr) , 128 , size(arr))
d(arr)
    return arr
end

function blobs_init(;arrtype=cu , SIZE=(256,1024) )
arr = arrtype(zeros(Float32,SIZE...))
dev = get_backend(arr)
#M = testdata(SIZE, 6, SIZE / 5, 2)
M = KernelAbstractions.zeros(dev , Float32 , size(arr) .- 2)
fill!(M , -1)
circle = set_circle(dev , 128  , size(M))
circle(M , 2f2 , CartesianIndex(1, 400))
circle(M , 1f2 , CartesianIndex(1, 800))
Inds = CartesianIndices(arr)
Id = one(Inds[begin])

arr[Inds[begin]+Id:Inds[end]-Id] = M
end
