using LinearAlgebra, Optim, Enzyme

# Constants needed for the model 

T = 50            # number of forward steps to take 
dt = .01         # timestep 

nt = Int(T / dt)

const g = 9.81          # gravity coefficient, meters^2 / second
const w_d = 2/3         # forcing frequency, 1 / second 
const b = 1.5           # radians / seconds^2 

const q_true = 100.0
const l_true = g 

# some functions that will help 
# forcing function, taken to be periodic

f(t) = b * cos( w_d * t )

# run one step of the nonlinear model, needs the inputs 
#           dt - timestep
#           k - current iteration 
#           f - forcing at the current step  
#           model - structure containing the state values  
function forward_step(dt, f, g, state_now, q, l)

    omega_new = 0.0
    theta_new = 0.0

    omega_new = (1 - dt/q) * state_now[1] - 
                (g * sin(state_now[2])/l + f) * dt

    theta_new = dt * state_now[1] + state_now[2]

    state_new = [copy(omega_new)
                 copy(theta_new)]
                 
    return state_new 

end

# build the "data" 

true_states = zeros(2, Int(T/dt))

for k = 2:Int(T / dt)

    state_new = forward_step(dt, f(k * dt), g, true_states[:, k-1], q_true, l_true)
    true_states[:, k] = state_new

end


# function for Enzyme (needs to return nothing)

function ad_forward(dt, f, g, q, l, state_now, state_new) 

    state_new = forward_step(dt, f, g, state_now, q, l)

    return nothing

end 

function ad_step(dt, f, g, q_guess, l_guess, state_now)
    
    d_state_now = zeros(2) 
    d_state_new = zeros(2)
    state_new = zeros(2) 

    autodiff(ad_forward, Const(dt), 
             Const(f), Const(g), 
             Const(q_guess), Const(l_guess), 
             Duplicated(state_now, d_state_now), Duplicated(state_new, d_state_new))

    return d_state_now, d_state_new 

    @show state_new 

end


