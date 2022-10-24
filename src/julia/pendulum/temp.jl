using Plots
T=500
include("pend_model.jl")
include("neural_net_one.jl")

@load "l_values.jld2" l_values 
#@load "q_values.jld2" q_values 

Ndata=100

variances = [2.0, 5.0, 10.0, 20.0, 50.0]

for n = 1:length(variances)
    this_variance = variances[n];
    local q_values = 100 .+ this_variance .* randn(1, Ndata)
    local diff_q = zeros(length(q_values), T+1)
    
    for j = 1:length(q_values)
    
        theta = generate_dataset(1, 500, 0.1, b, g, [0.0, 0.0], q_values[j], 9.81)
    
        diff_q[j, :] = theta[:] 
    
    end
    
    plot(1:T+1, diff_q[1:30,:]', title="q = 100 + " * string(this_variance) * "  .* rand(1, " * string(Ndata) * ")", legend=false)
    sleep(1.4)
    gui()
end
