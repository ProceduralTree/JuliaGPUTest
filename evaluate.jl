
    function evaluate(array , h::Float32; arrtype=T)

    device = get_backend(array)
    tmp = zeros(Float32, size(array)...) |> arrtype
    output = zeros(Float32, size(array)...) |> arrtype
    l = BoundaryKernels.left(device , 128 , size(array))
    grad  = gradient_add_kernel(device  , 128 , size(array))
    l(tmp)
    grad(array , output , h)
    sum(output .* tmp)
end
