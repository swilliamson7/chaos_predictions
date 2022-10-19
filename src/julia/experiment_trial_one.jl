using Flux, Statistics
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, logitcrossentropy, throttle, @epochs
using Base.Iterators: repeated
using Parameters: @with_kw
using CUDA
using MLDatasets

using JLD2, Random 

include("pend_model.jl")
include("netural_net_one.jl")

@load "l_values.jld2" l_values
@load "q_values.jld2" q_values 

Random.seed!(420)

Ndata = 10000

params = 100 .+ 0.01 .* randn(1, Ndata)

trajectories = generate_dataset(Ndata, 100, 0.1, b, g, [0.0;0.0], params, 9.81)

mutable struct Args
    n_train::Int      # number of training data
    n_test::Int         # number of test data
    n_validation::Int  # number of validation data
    η::Float64       # learning rate
    batchsize::Int    # batch size
    epochs::Int         # number of epochs
    device::Function   # set as gpu, if gpu available
end

train(trajectories, params, Args)