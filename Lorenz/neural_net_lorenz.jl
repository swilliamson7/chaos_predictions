using Flux, Statistics
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, logitcrossentropy, throttle, @epochs
using Flux.Losses: mse
using Base.Iterators: repeated
using Parameters: @with_kw
using CUDA
using MLDatasets
using JLD2 
using LinearAlgebra

include("create_structs.jl")
include("lorenz_model.jl")

# This basically checks if we are running on a GPU or not 
# Necessary for Flux to work 
if has_cuda()		# Check if CUDA is available
    @info "CUDA is on"
    CUDA.allowscalar(false)
end

# Generates a dataset (multiple trajectories with slightly perturbed parameter for each one)
# that can be separated into train, test, and validation. Needs to be given a structure
# generate_dataset_Args which contains 
#       N_data - the number of trajectories to create
#       T - how many timesteps to run each trajectory for (t = 0:dt:T)
#       dt - distance between timesteps
#       state0 - initial state for the system 
#       rho, sigma, beta - parameters for the Lorenz model
# One of rho, sigma, beta should be a row matrix of values, this will be the parameter
# that we want to learn to identify in the network 
function generate_dataset(s::generate_dataset_Args)

    # unpacks all of the values contained in the structure 
    @unpack_generate_dataset_Args s 

    # initializes storage for all of the trajectories, this is a three-dimensional
    # matrix 
    all_trajectories = zeros(3, T, N_data)
    
    # determine which parameter is perturbed by checking which matrix length exceeds 1
    param_strs = ["rho";"sigma";"beta"]
    is_perturbed = ([length(rho);length(sigma);length(beta)] .> (1,1,1)) 

    param_pert = param_strs[is_perturbed][1] 
  
    # this for loops runs over all of the perturbed parameters and for each one generates 
    # a trajectory of length T. They're all stored in the variable all_trajectories
    for k = 1:N_data
        if param_pert == "sigma"
            trajectory=generate_trajectory(T,
                                        dt,
                                        state0,
                                        rho[1],
                                        sigma[k],
                                        beta[1]
        )
            all_trajectories[:, :, k] = trajectory
        elseif param_pert == "rho" 
            trajectory=generate_trajectory(T,
                                        dt,
                                        state0,
                                        rho[k],
                                        sigma[1],
                                        beta[1]
        )
            all_trajectories[:, :, k] = trajectory          
        elseif param_pert == "beta" 
            trajectory=generate_trajectory(T,
                                        dt,
                                        state0,
                                        rho[1],
                                        sigma[1],
                                        beta[k]
        )
            all_trajectories[:, :, k] = trajectory
        end
     end

    return all_trajectories 
end

# This function is the exact same as the above, except that it takes in 
# individual inputs instead of a structure containing all of the inputs. 
# Not strictly necessary but nice because it offers us a choice of how we 
# want to generate our dataset 
function generate_dataset(N_data, T, dt, state0, rho, sigma, beta)
    all_trajectories = zeros(3, T, N_data)

    # determine which parameter is perturbed by checking which matrix length exceeds 1
    param_strs = ["rho";"sigma";"beta"]
    is_perturbed = ([length(rho);length(sigma);length(beta)] .> (1,1,1)) 
    param_pert = param_strs[is_perturbed][1] 

    for k = 1:N_data
        if param_pert == "sigma"
            trajectory=generate_trajectory(T,
                                        dt,
                                        state0,
                                        rho[1],
                                        sigma[k],
                                        beta[1]
        )
        all_trajectories[:, :, k] = trajectory
        elseif param_pert == "rho" 
            trajectory=generate_trajectory(T,
                                        dt,
                                        state0,
                                        rho[k],
                                        sigma[1],
                                        beta[1]
        )
        all_trajectories[:, :, k] = trajectory
        elseif param_pert == "beta" 
            trajectory=generate_trajectory(T,
                                        dt,
                                        state0,
                                        rho[1],
                                        sigma[1],
                                        beta[k]
        )
        all_trajectories[:, :, k] = trajectory
        end
    
     end

    return all_trajectories 

end

# After generating a bunch of different trajectories in generate_data we split them into the 
# canonical train, validate, test groups here. 
# Input: 
#       trajectories - the output of generate_data, so a bunch of different timeseries 
#       params - the row vector of perturbed parameters 
#       train_Args - a structure containing information about our neural net 
# Output: 
#       x_train - all trajectories that we want to train on 
#       y_train - parameters used to create x_train 
#       x_test - all trajectories we want to test on 
#       y_test - parameters that created the testing data
#       x_validation - trajectories that can be used for validation, say hyperparameter tuning
#       y_validation - parameters that were used to create x_validation 
function split_dataset(trajectories, params, Args::train_Args)

    x_train = trajectories[1:Args.n_train, :]
    x_test = trajectories[Args.n_train+1:Args.n_test+Args.n_train, :]
    x_validation = trajectories[Args.n_train+Args.n_test+1:end, :]

    y_train = params[1:Args.n_train]
    y_test = params[Args.n_train+1:Args.n_test+Args.n_train]
    y_validation = params[Args.n_train + Args.n_test + 1:end]

    return x_train, y_train, x_test, y_test, x_validation, y_validation

end

# This function takes the trajectories generated by generate_dataset and puts them 
# into DataLoader objects for Flux to then use with its functions. In general, this is just 
#       1. flattening the third dimension of our set of N_data trjacetories 
#       2. Parsing the trajectories into train, test, and validation
#       3. Pairing them with the parameter values and subsequently creating DataLoader objects 
# Input: 
#       trajectories - all of the data points generated with forward runs and slightly perturbed uncertain parameters
#       params - all of the parameters associated with the data points. should share the same indices (i.e. params[1]
#                generated trajectories[1])
#       Args - a structure containing information about the neural network set up, see definition for contents 
function batch_data(trajectories, params, Args)
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

    # flattening into a matrix whose columns are entire trajectories 
    flattened_trajectories = Flux.flatten(trajectories[:, :, :])

    # parsing data into train, test, and validation	
    x_train, y_train, x_test, y_test, x_validation, y_validation = split_dataset(flattened_trajectories', params, Args)
	
    # Reshape Data in order to flatten each image into a linear array
    x_train = x_train'
    x_test = x_test'

    # Batching
    train_data = DataLoader((x_train, y_train), batchsize=Args.batchsize, shuffle=true)
    test_data = DataLoader((x_test, y_test), batchsize=Args.batchsize)
    validation_data = DataLoader((x_validation, y_validation), batchsize=Args.batchsize)

    return train_data, test_data
end

# This is where we actually begin to build a neural net. This function is for if we want to run a 
# 2-layer NN, the first layer and second layer are both set to have 1000 nodes. Can easily 
# be modified to contain more layers, different numbers of nodes, etc.
function two_layer_model(trajectory_size; param_out=1)
    return Chain(
 	        Dense(trajectory_size, 1000, relu),
            Dense(1000, param_out)
            )
end

# Same as above except now we're using a single input layer and output layer model. 
# this is just linear regression as we're using the identity operator to take us from 
# input to output. For use with our ridge regression model 
function ridge_regression_model(trajectory_size; param_out=1)
    return Dense(trajectory_size, param_out, relu)
end

# This function computes various quantities comparing the predicted parameters 
# with the true parameters used to generate data points 
# Input: 
#       data_loader - the type of object that Flux stores the pairs (data, parameter) in
#       model -  which of the two model options to run, i.e. NN or ridge regression
#       device - this is an option that Flux has, if the computer we're running on 
#                has a gpu capable of being used for computation, can try and specify
#                here to run on the gpu, otherwise put cpu (we haven't tested this 
#                code on a gpu yet)
#       lambda - the hyperparameter used when computing the loss function
# Output:
#       loss - MSE loss
#       y??_vec - the predicted parameter values
#       acc - the accuracy of the predicted values, i.e. relative error
#       squared_error - MSE of predicted versus true
#       average - average parameter value
function score(data_loader, model, device)
    acc = 0
    average = 0
    loss = 0.0f0
    num = 0
    squared_error = 0.0
    ??_vec = Matrix{Float64}(undef, 1,0)
    for (x, y) in data_loader

        y = reshape(y, 1, length(y))
        x, y = device(x), device(y)
        ?? = model(x)

        squared_error = mse(??, y)
        average = average + sum(y) 

        loss += squared_error 
        num +=  size(x)[end]
        acc += norm(y??-y)/norm(y)
        
        y??_vec=[??_vec ??]

    end

    return loss / num, ??_vec, acc / num, squared_error, average / num 

end


# This function trains a neural network given a dataset of trajectories
# Input: 
#       trajectories - an array of Lorenz trajectories, each trajectory being an input
#                      layer to the neural network
#       params - a vector of Lorenz model parameters (sigma, rho, or beta) 
#                of interest. The neural network should, given a trajectory,
#                determine the corresponding value of the parameter.
#       args - a struct containing additional variables used in training (hyperparameters)
# Output:
#       {train,test}_data - the test and training data, respectively
#       predicted_params_{train,test} - the predicted parameters given train and
#                                       test data, respectively
#       {train,test}_acc_vec - vectors giving the accuracy of training and testing,
#                                       respectively
function train(trajectories, params, args)

    # if available use gpu, else use cpu
    if CUDA.functional() && args.use_cuda
        @info "Training on CUDA GPU"
        CUDA.allowscalar(false)
        device = gpu
    else
        @info "Training on CPU"
        device = cpu
    end

    # Partition data into train, test, and validation
    train_data, test_data = batch_data(trajectories, params, args)

    # Construct model
    model = args.model(length(train_data.data[1][:,1])) |> device
    model_params = Flux.params(model) ## model's trainable parameters

    # Training
    opt = ADAM(args.??)
		
    # initialize vectors to store output
    predicted_params_train = []
    predicted_params_test = []
    train_acc_vec = []
    test_acc_vec = []

    # loop through epochs
    for epoch in 1:args.epochs

        for (x, y) in train_data
            y = reshape(y, 1, length(y))
            x, y = device(x), device(y) ## transfer data to device
            gs = Flux.gradient(() -> loss(model(x), y), model_params) ## compute gradient
            Flux.Optimise.update!(opt, model_params, gs) ## update weights
        end
        

        # score model
        train_score, ??_vec_train, train_acc, _, _ = args.score(train_data, model, device)
        test_score, ??_vec_test, test_acc, _, _ = args.score(test_data, model, device)

        # append outputs
        push!(predicted_params_train, y??_vec_train)
        push!(predicted_params_test, y??_vec_test)
        push!(train_acc_vec, train_acc)
        push!(test_acc_vec, test_acc)

        println("Epoch=$epoch")
        println("Train score = $train_score, Train accuracy = $train_acc")
        println("Test score = $test_score, Test accuracy = $test_acc")

    end


    return train_data, test_data, predicted_params_train, predicted_params_test, train_acc_vec, test_acc_vec 

end

# This function performs ridge regression given a dataset of trajectories
# !!! Not currently running ridge regression, need to fix !!!
# Input: 
#       trajectories - an array of Lorenz trajectories, each trajectory being an input
#                      layer to the regression model
#       params - a vector of Lorenz model parameters (sigma, rho, or beta) 
#                of interest. The model should, given a trajectory, determine the 
#                corresponding value of the parameter.
#       args - a struct containing additional variables used in training (hyperparameters)
# Output:
#       {train,test}_data - the test and training data, respectively
#       predicted_params_{train,test} - the predicted parameters given train and
#                                       test data, respectively
#       {train,test}_acc_vec - vectors giving the accuracy of training and testing,
#                                       respectively
function train_RR(trajectories, params, args; lambda = .01)

    # if available use gpu, else use cpu
    if CUDA.functional() && args.use_cuda
        @info "Training on CUDA GPU"
        CUDA.allowscalar(false)
        device = gpu
    else
        @info "Training on CPU"
        device = cpu
    end

    # Partition data into train, test, and validation, currently not returning validation
    train_data, test_data, = batch_data(trajectories, params, args)

    # Construct model
    model = args.model(length(train_data.data[1][:,1])) |> device
    model_params = Flux.params(model)               # model's weights

    loss(x, y) = mse( x, y, agg=mean ) #+ lambda * sum(weights[1].^2)
 
    # Training
    opt = ADAM(args.??)
		
    # initialize vectors to store output
    predicted_params_train = []
    predicted_params_test = []
    train_acc_vec = []
    test_acc_vec = []

    # loop through epochs
    for epoch in 1:args.epochs

        for (x, y) in train_data
            y = reshape(y, 1, length(y))
            x, y = device(x), device(y) ## transfer data to device
            gs = Flux.gradient(() -> loss(model(x), y), model_params) ## compute gradient
            Flux.Optimise.update!(opt, model_params, gs) ## update parameters
        end

        # score model
        train_loss, ??_vec_train, train_acc, _, _ = args.score(train_data, model, device)
        test_loss, ??_vec_test, test_acc, _, _ = args.score(test_data, model, device)
        
        # append outputs
        push!(predicted_params_train, y??_vec_train)
        push!(predicted_params_test, y??_vec_test)
        push!(train_acc_vec, train_acc)
        push!(test_acc_vec, test_acc)

        println("Epoch=$epoch")
        println("Train score = $train_loss, Train accuracy = $train_acc")
        println("Test score = $test_loss, Test accuracy = $test_acc")

    end

    return train_data, test_data, predicted_params_train, predicted_params_test, train_acc_vec, test_acc_vec 

    
end
