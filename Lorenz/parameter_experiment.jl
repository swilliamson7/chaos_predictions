# This script includes all the necessary packages as well as other Julia scripts with 
# functions used for the model. Here is where a lot of the parameters in the model 
# can be altered, and the code is set up in such a way that this is the only place 
# necessary to change them. We do fix a random seed so that the random numbers generated 
# (noise in data, varying sigmas) is consistent across runs 

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
using LaTeXStrings

include("create_structs.jl")
include("lorenz_model.jl")
include("neural_net_lorenz.jl")
include("plotting.jl")

# initialize variables used to generate data
Random.seed!(420)
N_data = 7000
T = 500
dt = 0.001
state0=[1.0;0.0;0.0]
sigma = 10 .+ 5 .* randn(1, N_data)
rho = hcat(28.0)
beta = hcat(8/3)

# place data-generation variables in struct
generate_dataset_args = generate_dataset_Args(N_data=N_data, 
                                              T=T, 
                                              dt=dt, 
                                              state0=state0, 
                                              rho=rho, 
                                              sigma=sigma, 
                                              beta=beta
)

# generate trajectories data
trajectories = generate_dataset(generate_dataset_args)

# subset trajectories data and perturb
every_nth = 75
not_full_trajectories = trajectories[:, 1:every_nth:T, :] + 0.01 .* randn(3, length(1:every_nth:T), N_data)

# set variables used in training
epochs = 50
args = train_Args(Int(floor(0.8 * N_data)), 
                  Int(floor(0.1 * N_data)), 
                  Int(floor(0.1 * N_data)),  
                  3e-4, 
                  Int(floor(0.01 * N_data)), 
                  epochs, 
                  gpu,
                  ridge_regression_model,
                  score
)

# train (currently using ridge regression)
train_data, test_data, predicted_params_train, predicted_params_test, train_acc_vec, test_acc_vec = train_RR(not_full_trajectories, sigma, args)

# plot accuracy
pAccTest=plot_acc(epochs, test_acc_vec, "test accuracy")
pAccTrain=plot_acc(epochs, train_acc_vec, "train accuracy")
for_presentation = plot(pAccTest, pAccTrain, size=(600, 400))
