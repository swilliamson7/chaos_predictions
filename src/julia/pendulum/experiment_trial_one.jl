using Flux, Statistics
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, logitcrossentropy, throttle, @epochs
using Base.Iterators: repeated
using Parameters: @with_kw
using CUDA
using MLDatasets
using JLD2, Random, Plots
using Debugger

include("create_structs.jl")
include("dataset_utils.jl")
include("pend_model.jl")
include("neural_net_one.jl")

@load "l_values.jld2" l_values
@load "q_values.jld2" q_values 

Random.seed!(420)

N_data = 10000
params = 100 .+ 20.0 .* randn(1, N_data)
out_dir = "./experiment_trial_variance_50/"
dataset_filename = "dataset.jdl2"

if !isdir(out_dir)
    mkdir(out_dir)
end

# struct generate_dataset_Args loaded from create_structs.jl
generate_dataset_args = generate_dataset_Args(N_data, 500, 0.1, b, g, [0.0;0.0], params, 9.81)

# if dataset_filename exists in out_dir, load it. Else, create and save it.
trajectories = load_dataset(out_dir * dataset_filename, generate_dataset_args)

# set train args
args = train_Args(4000, 5000, 4000, 3e-4, 200, 10, gpu)
train_data, test_data, ŷ_vec_train, ŷ_vec_test = train(trajectories, params, args)

# plot
ŷ_vec_train = ŷ_vec_train'
ŷ_vec_test = ŷ_vec_test'

y_vec_train = train_data.data[2]
x=1:length(ŷ_vec_train)

plot(x, abs.(ŷ_vec_train' - y_vec_train), seriestype = :scatter, label = "Diff")

every_nth = 100
plot(x[1:every_nth:end], ŷ_vec_train[1:every_nth:end], seriestype = :scatter, label = "ŷ_vec_train") 
plot!(x[1:every_nth:end], y_vec_train[1:every_nth:end], seriestype = :scatter, label = "y_vec_train") 
