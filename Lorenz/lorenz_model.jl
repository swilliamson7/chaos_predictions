# building functions for the Lorenz model 

# run one step of the nonlinear model
# inputs: out - a shadow output, we copy to this the state after taking one timestep 
#         dt - timestep
#         x - the state at the current timestep
#         rho, sigma, beta - the Lorenz parameters
function forward_step(out, dt, x, rho, sigma, beta) 

    dx = zeros(3)

    dx[1] = sigma * (x[2] - x[1])
    dx[2] = x[1] * (rho - x[3]) - x[2]
    dx[3] = x[1] * x[2] - beta * x[3] 

    new = zeros(3)
    new[1] = x[1] + dt * dx[1]
    new[2] = x[2] + dt * dx[2] 
    new[3] = x[3] + dt * dx[3]

    copyto!(out, new)

end

# Creates a data point (timeseries) that we can use to train our neural network 
# Input: 
#           T - total steps to take 
#           dt - timestep 
#           state0 - initial state of the system 
#           params - structure containing the parameters 
# Output:
#           all_states - all the states of the system, from t = 1 to T 
function generate_trajectory(T, dt, state0, params)

    all_states = zeros(3, T)
    all_states[:, 1] = state0
    
    for j = 2:T
        out = zeros(3)
        forward_step(out, dt, all_states[:, j-1], params.rho, params.sigma, params.beta) 
        all_states[:, j] = out[:]
    end

    return all_states

end

# Same as above function except gives the option to feed individual parameters
# rather than a structure containing all of them 
function generate_trajectory(T, dt, state0, rho, sigma, beta)

    all_states = zeros(3, T)
    all_states[:, 1] = state0
    
    for j = 2:T
        out = zeros(3)
        forward_step(out, dt, all_states[:, j-1], rho, sigma, beta) 
        all_states[:, j] = out[:]
    end

    return all_states

end


# Function mainly for convenience, this will run one single adjoint 
# step for us. 
# Inputs: 
#       dt - timestep
#       state_now - state at the current iteration 
#       adjoint_old - the prior (formally future) adjoint value 
#       rho, sigma, beta - Lorenz parameters
# Outputs: 
#       d_state_now - the new adjoint variable 
function ad_step(dt, state_now, adjoint_old, rho, sigma, beta)
    
    adjoint_new = zeros(3) 
    output_state = zeros(3)
    autodiff(forward_step, Const,
            Duplicated(output_state, adjoint_old),
            Const(dt), 
            Duplicated(state_now, adjoint_new), 
            Const(rho),
            Const(sigma),
            Const(beta)
    )

    return adjoint_new 

end

# This function will put together all of the above and run the entire 
# forwards-backwards part of the adjoint problem. 
# Inputs: 
#       data_steps - the steps which will incorporate data 
#       data - the data values themselves
#       dt - timestep
#       T - final time
#       state0 - initial state of the system 
#       params - a structure containing the parameters in the Lorenz model, 
#                rho, sigma, and beta 
# Outputs: 
#       states - all of the results from the forward run with the 
#                inputted parameters
#       adjoint_variables - the adjoint variables (Lagrange multipliers)
#                that were computed 
function adjoint(data_steps, data, dt, T, state0, rho, sigma, beta)

states = generate_trajectory(T, dt, state0, rho, sigma, beta)

# next we want to run the adjoint problem backward and compute 
# the adjoint variables 
adjoint_variables = zeros(3, T)

for j = T-1:-1:1

    adjoint_old = adjoint_variables[:, j+1]
    d_state_now = ad_step(dt, states[:,j], adjoint_old, rho, sigma, beta)
    adjoint_variables[:,j] .= d_state_now[:]

    # this statement checks if we have a data point at the current iteration, and if so adjusts
    # the adjoint value 

    ## This needs to be fixed to work with the Lorenz model, currently still set up for the pendulum ##################
    if j in data_steps

        adjoint_variables[2,j] = adjoint_variables[2,j] + 1/(length(data_steps)) * (states[2,j] - data[2,j])^2 

    end

end

return states, adjoint_variables 

end