using Parameters

@with_kw mutable struct generate_dataset_Args
    N_data::Int64
    T::Int64
    dt::Float64
    b::Matrix{Float64}
    g::Float64
    state0::Vector{Float64}
    q::Matrix{Float64}
    l::Matrix{Float64}
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
