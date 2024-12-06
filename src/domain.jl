 ; using Plots
using LaTeXStrings
pgfplotsx()
Idx = CartesianIndex(1,1)
M = zeros(66,66)
M[2:end-1 , 2:end-1] = ones(64,64)
heatmap(M, title=L"\Omega_d" , clim=(0,1),
            gridlinewidth=2 , axis_equal_image=true , extra_kwargs=:subplot , xlims=(1 ,66) , ylims=(1,66), xlabel=L"x_1",ylabel=L"x_2");
