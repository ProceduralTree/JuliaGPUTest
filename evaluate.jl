
function evaluate(array::Array ; arrtype=T)

    device = get_backend(initialCondition)
    tmp = zeros(Float32, size(initialCondition)...) |> arrtype
    output = zeros(Float32, size(initialCondition)...) |> arrtype
    l = BoundaryKernels.left(device , 128 , size(initialCondition))
    div  = gdiv(device  , 128 , size(array))
    l(tmp)
    div(array , output)
    sum(output .* tmp)
end
