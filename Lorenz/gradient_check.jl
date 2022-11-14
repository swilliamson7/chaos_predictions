# checking that the gradient we're computing is correct, following what Omar did in the Catenary problem 
# done via computing a directional derivative two different ways 

using Random, Enzyme, LinearAlgebra, Plots

include("lorenz_model.jl")

# set the random seed 
Random.seed!(420)

# total time steps to run 
T = 100
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
data = all_states_true[:, data_steps] + 0.01 .* randn(3, length(data_steps))

x1 = 0.1 * randn(1)[1] + 10;
p = randn(1)[1] + 10;
step_sizes = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13]
# #s = 1e-7

all_states_adjoint, adjoint_variables = adjoint(data_steps, data, dt, T, state0, rho_true, x1, beta_true)

J1 = data_misfit(all_states_adjoint, all_states_true, data_steps)
g = gradient(adjoint_variables, all_states_adjoint, T)
dJ_dx_p = dot(g, p)

grad_errs = []

for s in step_sizes

    all_states_adjoint_new = []
    adjoint_variables_new = []

    x2 = x1 + s*p
    all_states_adjoint_new, adjoint_variables_new = adjoint(data_steps, data, dt, T, state0, rho_true, x2, beta_true)
    J2 = data_misfit(all_states_adjoint_new, data, data_steps)
    dJ_dx_p_diff = (J2 - J1) / s
    
    error = abs( (dJ_dx_p - dJ_dx_p_diff) / dJ_dx_p_diff )
    push!(grad_errs, error)
end

plot(step_sizes, grad_errs, xaxis=:log, yaxis=:log)