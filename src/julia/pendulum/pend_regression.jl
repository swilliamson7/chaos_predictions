using Plots
using Flux, Statistics
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, logitcrossentropy, throttle, @epochs
using Base.Iterators: repeated
using Parameters: @with_kw
using CUDA
using MLDatasets
using JLD2, Random, Plots
using Plots.PlotMeasures
using Debugger
using LaTeXStrings

include("create_structs.jl")
include("dataset_utils.jl")
include("pend_model.jl")
include("neural_net_one.jl")
include("plotting.jl")

Random.seed!(420)

N_data = 10000
b_perturb_vec = 1.5 .+ 5 .* randn(1, N_data)
out_dir = "./experiment_trial_variance_perturb_b/"
dataset_filename = "dataset.jdl2"
lambdas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]

loss(x,y) = Flux.mse(model(x), y)

# struct generate_dataset_Args loaded from create_structs.jl
nT = 500
generate_dataset_args = generate_dataset_Args(N_data=N_data, T=nT, dt=0.1, 
            b=b_perturb_vec, g=g, state0=[0.0;0.0], q=hcat(100), l=hcat(9.81))


args = train_Args(Int(floor(0.7 * N_data)), Int(floor(0.1 * N_data)), Int(floor(0.2 * N_data)), 3e-4, 10, 1, gpu)

trajectories = generate_dataset(generate_dataset_args);

train_data, test_data, validation_data = getdata(trajectories, b_perturb_vec, args)

model = regression_model()

ŷ_vec_validation, validation_losses, stuff = hyperparameter_training(trajectories, b_perturb_vec, args, lambdas)

plot(lambdas, stuff, xaxis=:log)
ylim([0,1])