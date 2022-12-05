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
Random.seed!(420)           # fixes a random seed

# This function will be what runs the NN given a set of inputs. To be run it needs:
#           N_data - the total number of trajectories to create (will be batched into test and train, later need to add validation)
#           T - how many steps to integrate each trajectory for 
#           dt - timestep for the model, should be sufficiently small for stability 
#           state0 - initial state for the model
#           every_nth - which steps to take for data (i.e., if 10 then only knowledge of every 10th point is assumed for the NN)
#           sigma - model parameter sigma 
#           rho - model parameter rho 
#           beta - model parameter beta 
#           perturbed_param_string - which parameter we want the NN to try and predict, needs to be given as a string (i.e., "rho" or "sigma")
#           epochs - how many epochs to run 
#
# An example usage of the function is below. At the end of the run a plot will be given of the relative error of your chosen parameter on both the 
# test and train data  
#
# N_data = 7000               # determines how many trajectories to generate
# T = 500                     # how long to integrate the model 
# dt = 0.001                  # dt 
# state0=[1.0;0.0;0.0]        # initial value for the trajectories
# epochs = 5
# sigma=10.0
# rho=28.0
# beta=8/3
# perturbed_param_string="beta"
# every_nth = 75

# train_data, test_data, predicted_params_train, predicted_params_test, train_acc_vec, test_acc_vec = parameter_experiment(N_data, 
#                                                                                                                          T, 
#                                                                                                                          dt, 
#                                                                                                                          state0, 
#                                                                                                                          every_nth,
#                                                                                                                          sigma, 
#                                                                                                                          rho, 
#                                                                                                                          beta, 
#                                                                                                                          perturbed_param_string, 
#                                                                                                                          epochs 
# )

# # plot accuracy
# pAccTest=plot_acc(epochs, test_acc_vec, "test accuracy")
# pAccTrain=plot_acc(epochs, train_acc_vec, "train accuracy")
# plot(pAccTest, pAccTrain, size=(600, 400))

function parameter_experiment(N_data, T, dt, state0, every_nth, sigma, rho, beta, perturbed_param_string::String, epochs)

    # Picking parameters. If we want to run the same test with rho 
    # and not sigma, just change which one is a vector and which is held constant. 
    # Currently the code is not set up to handle multiple choices at once 

    sigma = hcat(sigma) 
    rho = hcat(rho) 
    beta = hcat(beta)

    if perturbed_param_string == "sigma"
        sigma = sigma .+ (0.5 * abs(sigma[1])) .* randn(1, N_data)
        perturbed_param=sigma
    elseif perturbed_param_string == "rho" 
        rho = rho .+ (0.5 * abs(rho[1])) .* randn(1, N_data)
        perturbed_param=rho
    elseif perturbed_param_string == "beta" 
        beta = beta .+ (0.5 * abs(beta[1])) .* randn(1, N_data)
        perturbed_param=beta
    end

    @show perturbed_param[1:10]


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
    every_nth = every_nth
    not_full_trajectories = trajectories[:, 1:every_nth:T, :] + 0.01 .* randn(3, length(1:every_nth:T), N_data)

    # set variables used in training

    model = ridge_regression_model
    args = train_Args(Int(floor(0.8 * N_data)), 
                    Int(floor(0.1 * N_data)), 
                    Int(floor(0.1 * N_data)),  
                    3e-4, 
                    Int(floor(0.01 * N_data)), 
                    epochs, 
                    gpu,
                    model,
                    score
    )

    # train (not currently using ridge regression)
    train_data, test_data, predicted_params_train, predicted_params_test, train_acc_vec, test_acc_vec = train_RR(not_full_trajectories, 
                                                                                                                perturbed_param, 
                                                                                                                args
    )

    return train_data, test_data, predicted_params_train, predicted_params_test, train_acc_vec, test_acc_vec

end 
