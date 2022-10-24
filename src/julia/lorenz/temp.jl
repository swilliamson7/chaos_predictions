using Plots

include("lorenz_model.jl")

params = lorenz_params(28, 10, 8/3)
T = 1000

all_states = zeros(3, T)
all_states[:, 1] = [1.0, 0.0, 0.0]

for j = 2:T
    out = zeros(3)
    forward_step(out, 0.01, all_states[:, j-1], params.rho, params.sigma, params.beta) 
    all_states[:, j] = out
end

plot(all_states[1,:], all_states[2,:],all_states[3,:])