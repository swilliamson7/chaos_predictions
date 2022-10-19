using JLD2, Random

q_values = 100 .+ 0.001 .* randn(1, 10000)
l_values = 9.81 .+ 0.001 .* randn(1, 10000)

@save "q_values.jld2" q_values 

@save "l_values.jld2" l_values

@load 