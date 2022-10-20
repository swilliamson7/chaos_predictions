
include("pend_model.jl")
include("neural_net_one.jl")

@load "l_values.jld2" l_values 

diff_l = zeros(length(l_values), T+1)

for j = 1:length(l_values)

    theta = generate_dataset(1, 500, 0.1, b, g, [0.0, 0.0], q_values[j], l_values[j])

    diff_l[j, :] = theta[:] 

end