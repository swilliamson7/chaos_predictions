# This script will be a test, my goal is to just (a) define a structure
# for the pendulum that will contain fields necessary to the model
# (b) define functions that we'll use (such as a forward step)
# and (c) check the adjoint variables 

mutable struct pendulum 

    # # "known" parameters 
    T::Int64            # total steps to take 
    dt::Float64         # time step

    g::Float64          # gravity coefficient
    w_d::Float64        # 
    b::Float64          #

    # components of our state vector 

    omega::Vector{Float64} 
    theta::Vector{Float64} 

    # "unknown" parameters 

    q::Float64          # damping coefficient, seconds
    l::Float64          # length of the pendulum, meters

end

# forcing function, taken to be periodic

f(t) = b * cos( w_d * t )

# initializes the fields in our structure, i.e. just putting vectors of zero

# function allocate_memory(T, dt, g, w_d, b, q, l)

#     g = g 
#     w_d = w_d
#     b = b

#     nt = T / dt
    
#     omega = zeros(Int(nt))
#     theta = zeros(Int(nt))

#     model = pendulum(omega, theta, q, l)

#     return model 

# end

# run one step of the nonlinear model, needs the inputs 
#           dt - timestep
#           k - current iteration 
#           f - forcing at the current step  
#           model - structure containing the state values  
function forward_step(dt, k, f, model)

    omega_new = 0.0
    theta_new = 0.0

    omega_new = (1 - dt/model.q) * model.omega[k-1] - 
                (model.g * sin(model.theta[k-1])/model.l + f) * dt

    theta_new = dt * model.omega[k-1] + model.theta[k-1]

    model.omega[k] = copy(omega_new)
    model.theta[k] = copy(theta_new) 

end



