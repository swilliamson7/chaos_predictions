using Plots

include("lorenz_model.jl")

rho = 28.0
sigma = 10.0
beta = 8/3

T = 10

all_states = zeros(3, T)
all_states[:, 1] = [1.0, 0.0, 0.0]

for j = 2:T
    out = zeros(3)
    forward_step(out, 0.01, all_states[:, j-1], rho, sigma, beta) 
    all_states[:, j] = out
end

plot(all_states[1,:], all_states[2,:],all_states[3,:], label = "", title = "10 steps", xlabel = "x", ylabel = "y", zlabel = "z")
# xlab!("x")
# ylab="y"
# zlab="z"