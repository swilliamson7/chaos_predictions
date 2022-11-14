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
include("lorenz_model.jl")
include("neural_net_lorenz.jl")
#include("conv.jl")
include("plotting.jl")


Random.seed!(420)
N_data=100000
lambda = 0.01
sigma = 10 .+ 5 .* randn(1, N_data)
rho = hcat(28.0)
beta = hcat(8/3)

# struct generate_dataset_Args loaded from create_structs.jl

generate_dataset_args = generate_dataset_Args(N_data=N_data, 
                                              T=500, 
                                              dt=0.001, 
                                              state0=[1.0;0.0;0.0], 
                                              rho=rho, 
                                              sigma=sigma, 
                                              beta=beta
                                            )

# if dataset_filename exists in out_dir, load it. Else, create and save it.
trajectories = generate_dataset(generate_dataset_args)
T = generate_dataset_args.T
every_nth = 75
not_full_trajectories = trajectories[:, 1:every_nth:T, :] + 0.01 .* randn(3, length(1:every_nth:T), N_data)

# set train args and train
epochs=10

args = train_Args(Int(floor(0.8 * N_data)), 
                  Int(floor(0.1 * N_data)), 
                  Int(floor(0.1 * N_data)),  
                  3e-4, 
                  150, 
                  epochs, 
                  gpu,
                  ridged_regression_model,
                  ridge_regression_loss)


train_data, test_data, ŷ_vec_train, ŷ_vec_test, train_acc_vec, test_acc_vec = train(not_full_trajectories, sigma, args)

# plot some predicted versus true parameters output from training
ŷ_vec_train = ŷ_vec_train'
ŷ_vec_test = ŷ_vec_test'
y_vec_train = train_data.data[2]
y_vec_test = test_data.data[2]
x=1:length(ŷ_vec_train)

# here we plot training data
pTrain=plot_data_pred_vs_true(100, ŷ_vec_train, y_vec_train, pred_label="ŷ_vec_train", true_label="y_vec_train")
pTest=plot_data_pred_vs_true(100, ŷ_vec_test, y_vec_test, pred_label="ŷ_vec_test", true_label="y_vec_test")
plot(pTrain, pTest,layout=(2,1), size=(600, 400))

# plot(x, abs(ŷ_vec_train - y_vec_train))


#plot_data_pred_minus_true(1000, pred_vec, true_vec)

# # plot accuracy
# pAccTest=plot_acc(epochs, test_acc_vec, "test accuracy")
# pAccTrain=plot_acc(epochs, train_acc_vec, "train accuracy")
# plot(pTrain, pTest, pAccTest, pAccTrain, layout=(2,2), size=(600, 400))
