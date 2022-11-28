# This script creates a plot of integration time versus sigma, where sigma 
# is computed via the adjoint method. 

using Optim, Enzyme, Random, Plots, LaTeXStrings

# all necessary functions are in this script
include("lorenz_model.jl")

# set the random seed 
Random.seed!(420)

# total time steps to run 
Ts = [100 + 10*k for k = 0:21]

sigmas = []

for T in Ts

    # initial condition
    state0 = [1.0, 0.0, 0.0]

    # setting the parameters that we'll consider the "true" values 
    rho_true = 28.0
    sigma_true = 10.0 
    beta_true = 8/3

    # we assume we have near perfect observation at every nth timestep, the value set here
    every_nth = 1
    data_steps = 1:every_nth:T

    # generate the data for the adjoint method 
    all_states_true = generate_trajectory(T, 0.01, state0, rho_true, sigma_true, beta_true)
    data = all_states_true[:, data_steps] + 0.1 .* randn(3, length(data_steps))

    sigma = grad_descent(10.3, 50, data_steps, data, 0.01, T, state0, 28.0, 8/3)

    push!(sigmas, sigma)

end

plot(Ts, sigmas, seriestype = :scatter, label = "", xlabel="Integration time", ylabel=L"\sigma", dpi = 300)