# This script creates structures that will be used within neural_net_lorenz.jl
# Not strictly necessary to have the structures, but Julia performs best when we don't use 
# global variables so it's good practice 

using Parameters

# Contains variables relating to the forward run. 
#       N_data - the number of data points (trajectories) to generate
#       T - the length of integration (how many steps to run the model for)
#       dt - timestep 
#       state0 - the initial state of the Lorenz model 
#       rho, sigma, beta - the Lorenz parameters. These will be considered the true values 
@with_kw mutable struct generate_dataset_Args
    N_data::Int64
    T::Int64
    dt::Float64
    state0::Vector{Float64}
    rho::Matrix{Float64}
    sigma::Matrix{Float64}
    beta::Matrix{Float64}
end

# Contains variables relating to the ML algorithm
#       n_train - number of data points to allocate to training
#       n_test - number of data points to allocate to test
#       n_validation - number of data points to allocate to validation 
#       η - the step size for our optimization algorithm
#       batchsize - as it sounds, the batch size for train/test/validation data
#       epochs - number of epochs to train for 
#       device - a Flux variable, decides if we want to train on gpu/cpu (for our purposes cpu)
#       model - which of the ML models we want to use (currently between a standard NN or ridge regression)
#       score - scores the results by returning a few values (MSE, relative error, etc.)
# For more information on the last two functions see neural_net_lorenz.jl, the functions are defined in 
# that script. 
@with_kw mutable struct train_Args
    n_train::Int         # number of training data
    n_test::Int          # number of test data
    n_validation::Int    # number of validation data
    η::Float64           # step size 
    batchsize::Int       # batch size
    epochs::Int          # number of epochs
    device::Function     # set as gpu, if gpu available
    model::Function      # model
    score::Function      # checks the accuracy of the model 
end


mutable struct Lorenz_parameters 

    rho::Float64
    sigma::Float64
    beta::Float64

end
