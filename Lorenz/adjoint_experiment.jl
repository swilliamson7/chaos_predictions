# This script creates a plot of integration time versus sigma, where sigma 
# is computed via the adjoint method. 

using Optim, Enzyme, Random, Plots, LaTeXStrings

# all necessary functions are in this script
include("lorenz_model.jl")

# Will generate a plot of integration time versus estimated sigma. Currently only 
# set up to compute the gradient with respect to sigma. Needs as input:
#                   state0 - initial state for the Lorenz model 
#                   rho - true (fixed) rho value 
#                   sigma - true (fixed) sigma value 
#                   sigma_guess - the initial guess for sigma, to be given to gradient descent 
#                   beta - true (fixed) beta value 
#                   every_nth - which steps will be used as data for the adjoint method (1:every_nth:T
#                               for T the integration time)
#                   Ts - What integration times to run over 
# Returns:
#                   sigmas - all of the sigma values that adjoint + gradient descent computed 
#
# Example usage:
#
# include("adjoint_experiment.jl") 
# state0 = [1.0, 0.0, 0.0]
# rho = 28.0
# sigma = 10.0
# sigma_guess = 12.0 
# beta = 8/3 
# every_nth = 20 
# Ts = 100
# sigmas = adjoint_experiment(state0, rho, sigma, sigma_guess, beta, every_nth, Ts)
# plot(Ts, sigmas, seriestype = :scatter, label = "", xlabel="Integration time", ylabel=L"\sigma", dpi = 300)
function adjoint_experiment(state0, rho, sigma, sigma_guess, beta, every_nth, Ts)

    # set the random seed 
    Random.seed!(420)

    # total time steps to run 
    Ts = Ts

    sigmas = []

    for T in Ts

        # initial condition
        state0 = state0

        # setting the parameters that we'll consider the "true" values 
        rho_true = rho
        sigma_true = sigma
        beta_true = beta

        # we assume we have near perfect observation at every nth timestep, the value set here
        every_nth = every_nth
        data_steps = 1:every_nth:T

        # generate the data for the adjoint method 
        all_states_true = generate_trajectory(T, 0.01, state0, rho_true, sigma_true, beta_true)
        data = all_states_true[:, data_steps] + 0.1 .* randn(3, length(data_steps))

        sigma = grad_descent(sigma_guess, 75, data_steps, data, 0.01, T, state0, rho, beta)

        push!(sigmas, sigma)

    end

    return sigmas 

end

