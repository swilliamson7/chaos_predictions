using Optim, Enzyme, Random 

include("lorenz_model.jl")

Random.seed!(420)

T = 100 
dt = 0.01 
state0 = [1.0, 0.0, 0.0]

rho_true = 28.0

sigma_true = 10.0 

beta_true = 8/3

every_nth = 30

all_states_true = generate_trajectory(T, dt, state0, rho_true, sigma_true, beta_true)

data = all_states_true[:, 1:every_nth:T] + 0.01 .* randn(3, length(1:every_nth:T))

all_states_adjoint, adjoint_variables = adjoint(1:every_nth:T, data, dt, T, state0, rho_true, sigma_true + .1, beta_true)