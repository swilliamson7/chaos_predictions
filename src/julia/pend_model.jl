# fixed parameters of the system 

const g = 9.81          # gravity coefficient, meters^2 / second
const w_d = 2/3         # forcing frequency, 1 / second 
const b = 1.5           # radians / seconds^2 

const q_true = 100.0
const l_true = g 

# building the functions for the model 

# forcing function, taken to be periodic
f(t) = b * cos( w_d * t )

# run one step of the nonlinear model, needs the inputs 
#           dt - timestep
#           k - current iteration 
#           g - gravitational acceleration  
#           state_now - state vector at the current iteration 
#           q - damping coefficient 
#           l - pendulum length 
function forward_step(dt, k, b, w_d, g, state_now, q, l)

    omega_new = 0.0
    theta_new = 0.0

    omega_new = (1 - dt/q) * state_now[1] - 
                (g * sin(state_now[2])/l + b * cos(w_d * dt * k)) * dt 

    theta_new = dt * state_now[1] + state_now[2]

    state_new = [copy(omega_new)
                 copy(theta_new)]
                 
    return state_new 

end

# Creates data that we can use to train our neural network, 
# it returns one specific trajectory with a given q and l and returns 
# what that trajectory was at every step. 
# Input: 
#           T - total steps to take 
#           dt - timestep 
#           f - forcing function 
#           g - gravitational acceleration
#           state0 - initial state of the system 
#           q - damping coefficient
#           l - length of the pendulum 
# Output:
#           all_states - all the states of the system, from t = 1 to T 
function generate_trajectory(T, dt, b, g, state0, q, l)

    all_states = zeros(2, T+1)
    all_states[:, 1] = state0

    for j = 2:T+1

        state_new = forward_step(dt, j, b, w_d, g, all_states[:, j-1], q, l)
        all_states[:, j] = state_new[:]

    end

    return all_states

end

# Function for Enzyme (needs to return nothing)
# just runs the above forward step function, but doesn't
# return the output, will be what we apply Enzyme to. 
# Inputs: 
#       dt - timestep
#       f - forcing *value* at the current iteration 
#       g - gravitational acceleration  
#       q - damping coefficient 
#       l - pendulum length 
#       state_now - state at the current iteration 
#       state_new - output of the forward function, given the above
#                   inputs 
# Outputs: nothing 
function ad_forward(dt, k, b, w_d, g, state_now, q, l, state_new) 

    state_new = forward_step(dt, k, b, w_d, g, state_now, q, l)

    return nothing

end 


# Function mainly for convenience, this will run one single adjoint 
# step for us. 
# Inputs: 
#       dt - timestep
#       g - gravitational acceleration
#       q_guess - damping coefficient, labelled a guess as 
#                   it's taken to be an unknown parameter 
#       l_guess - pendulum length, same as above 
#       state_now - state at the current iteration 
#       d_state_new - the prior (formally future) adjoint value 
# Outputs: 
#       d_state_now - the new adjoint variable 
function ad_step(dt, k, b, w_d, g, state_now, q_guess, l_guess, d_state_new)
    
    d_state_now = zeros(2) 

    autodiff(ad_forward, dt, k,
             b, w_d, g, 
             Duplicated(state_now, d_state_now), 
             q_guess, l_guess, 
             Duplicated(zeros(2), d_state_new))

    return d_state_now

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
function adjoint(data_steps, data, dt, b, w_d, T, g, state0, q0, l0)

# first run the entire forward problem 
states = zeros(2, Int(T/dt))
states[:, 1] = state0

for j = 2:Int(T/dt)
    state_new = forward_step(dt, j, b, w_d, g, states[:, j-1], q0, l0)
    states[:, j] = state_new 
end

# next we want to run the adjoint problem backward and compute 
# the adjoint variables 
adjoint_variables = zeros(2, Int(T/dt))

for j = Int(T/dt)-1:-1:1

    adjoint_old = adjoint_variables[:, j+1]
    d_state_now = ad_step(dt, j, b, w_d, g, states[:,j], q0, l0, adjoint_old)
    adjoint_variables[:,j] .= d_state_now[:]

    # this statement checks if we have a data point at the current iteration, and if so adjusts
    # the adjoint value 
    if j in data_steps

        adjoint_variables[2,j] = adjoint_variables[2,j] + 1/(length(data_steps)) * (states[2,j] - data[2,j])^2 

    end

end

return states, adjoint_variables 

end