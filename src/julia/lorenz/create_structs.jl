using Parameters

@with_kw mutable struct generate_dataset_Args
    Ndata::Int64
    T::Int64
    dt::Float64
    state0::Vector{Float64}

    rho::Float64
    sigma::Float64 
    beta::Float64

end

@with_kw mutable struct train_Args
    n_train::Int         # number of training data
    n_test::Int          # number of test data
    n_validation::Int    # number of validation data
    η::Float64           # step size 
    batchsize::Int       # batch size
    epochs::Int          # number of epochs
    device::Function     # set as gpu, if gpu available
end