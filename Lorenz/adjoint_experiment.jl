using Optim, Enzyme, Random 

# all necessary functions are in this script
include("lorenz_model.jl")

# set the random seed 
Random.seed!(420)

# total time steps to run 
T = 10
dt = 0.01 

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
all_states_true = generate_trajectory(T, dt, state0, rho_true, sigma_true, beta_true)
data = all_states_true[:, data_steps] #+ 0.01 .* randn(3, length(data_steps))

# incorporate the above with the adjoint method, assuming that we have perfect knowledge of the initial 
# condition, rho, and beta, but an imperfect sigma (about 10% too large)
# all_states_adjoint, adjoint_variables = adjoint(data_steps, data, dt, T, state0, rho_true, sigma_true + .1, beta_true)

M = 500
sigma0 = 12


sigma = grad_descent(sigma0, M, data_steps, data, dt, T, state0, 28.0, 8/3)
