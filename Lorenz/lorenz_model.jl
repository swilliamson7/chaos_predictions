# parameter struct 

mutable struct lorenz_params 
    rho::Float64
    sigma::Float64
    beta::Float64
end 

# building functions for the Lorenz model 

# run one step of the nonlinear model
# inputs: out - a shadow output, we copy to this the state after taking one timestep 
#         dt - timestep
#         x - the state at the current timestep
#         params - structure containing rho, sigma, beta 
function forward_step(out, dt, x, rho, sigma, beta) 

    dx = zeros(3)

    dx[1] = sigma * (x[2] - x[1])
    dx[2] = x[1] * (rho - x[3]) - x[2]
    dx[3] = x[1] * x[2] - beta * x[3] 

    new = x + dt .* dx 
    copyto!(out, new)

end

# Creates data that we can use to train our neural network, 
# it returns one specific trajectory with a given q and l and returns 
# what that trajectory was at every step. 
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
        all_states[:, j] = out
    end

    return all_states

end

function generate_trajectory(T, dt, state0, rho, sigma, beta)

    all_states = zeros(3, T)
    all_states[:, 1] = state0
    
    for j = 2:T
        out = zeros(3)
        forward_step(out, dt, all_states[:, j-1], rho, sigma, beta) 
        all_states[:, j] = out
    end

    return all_states

end


# Function mainly for convenience, this will run one single adjoint 
# step for us. 
# Inputs: 
#       dt - timestep
#       state_now - state at the current iteration 
#       adjoint_old - the prior (formally future) adjoint value 
# Outputs: 
#       d_state_now - the new adjoint variable 
function ad_step(dt, state_now, adjoint_old, params)
    
    adjoint_new = zeros(2) 
    drho, dsigma, dbeta = autodiff(forward_step, 
            Duplicated(state_now, adjoint_new),
            dt, 
            Duplicated(zeros(2), adjoint_old), 
            Active(params.rho),
            Active(params.sigma),
            Active(params.beta)
            )

    return drho, dsigma, dbeta, adjoint_new 

end

# This function will put together all of the above and run the entire 
# forwards-backwards part of the adjoint problem. 
# Inputs: 
#       data_steps - the steps which will incorporate data 
#       data - the data values themselves
#       dt - timestep
#       T - final time
#       f - forcing *function*
#       g - gravitational acceleration
#       state0 - initial state of the system 
#       q0 - initial guess for q, the damping coefficient
#       l0 - initial guess for l, the pendulum length 
# Outputs: 
#       states - all of the results from the forward run with the 
#                inputted parameters
#       adjoint_variables - the adjoint variables (Lagrange multipliers)
#                that were computed 
function adjoint(data_steps, data, dt, T, state0, params)

states = generate_trajectory(T, dt, state0, params)

# next we want to run the adjoint problem backward and compute 
# the adjoint variables 
adjoint_variables = zeros(2, Int(T/dt))

for j = Int(T/dt)-1:-1:1

    adjoint_old = adjoint_variables[:, j+1]
    d_state_now = ad_step(dt, states[:,j], adjoint_old, params)
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