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

N_data = 100000
b_perturb_vec = 1.5 .+ 0.1 .* randn(1, N_data)
out_dir = "./experiment_trial_variance_perturb_b/"
dataset_filename = "dataset.jdl2"

if !isdir(out_dir)
    mkdir(out_dir)
end

# struct generate_dataset_Args loaded from create_structs.jl
nT = 500
generate_dataset_args = generate_dataset_Args(N_data=N_data, T=nT, dt=0.1, b=b_perturb_vec, g=g, state0=[0.0;0.0], q=hcat(100), l=hcat(9.81))

# if dataset_filename exists in out_dir, load it. Else, create and save it.
trajectories = generate_dataset_perturb_b(generate_dataset_args)

# plot a few trajectories
pTraj=plot(1:nT+1, trajectories[1:30,:]', title= string(nT) * " " * L"\theta" * " trajectories", label=false, xlabel="t [seconds]", ylabel=L"\theta(t)") 

# set train args and train
epochs=10
args = train_Args(80000, 10000, 10000, 3e-4, 2000, epochs, gpu)
train_data, test_data, ŷ_vec_train, ŷ_vec_test, train_acc_vec, test_acc_vec = train(trajectories, b_perturb_vec, args)

# plot some predicted versus true parameters output from training
ŷ_vec_train = ŷ_vec_train'
ŷ_vec_test = ŷ_vec_test'
y_vec_train = train_data.data[2]
y_vec_test = test_data.data[2]
#x=1:length(ŷ_vec_train)

# here we plot training data
#pTrain=plot_data_pred_vs_true(100, ŷ_vec_train, y_vec_train, pred_label="ŷ_vec_train", true_label="y_vec_train")
#pTest=plot_data_pred_vs_true(100, ŷ_vec_test, y_vec_test, pred_label="ŷ_vec_test", true_label="y_vec_test")
#plot(pTrain, pTest, pAccTest, pAccTrain, layout=(2,1), size=(600, 400))


plot_data_pred_minus_true(1000, pred_vec, true_vec)

# plot accuracy
#pAccTest=plot_acc(epochs, test_acc_vec, "test accuracy")
#pAccTrain=plot_acc(epochs, train_acc_vec, "train accuracy")
#plot(pTrain, pTest, pAccTest, pAccTrain, layout=(2,2), size=(600, 400))



