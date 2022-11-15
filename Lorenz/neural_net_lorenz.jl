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

    # determine which parameter is preserved by checking which matrix length exceeds 1
    param_strs = ["rho";"sigma";"beta"]
    is_perturbed = ([length(rho);length(sigma);length(beta)] .> (1,1,1)) 
    param_pert = param_strs[is_perturbed][1] 
    param_unpert1 = param_strs[.!is_perturbed][1]
    param_unpert2 = param_strs[.!is_perturbed][2]

    # this for loops runs over all of the perturbed parameters and for each one generates 
    # a trajectory of length T. They're all stored in the variable all_trajectories
    for k = 1:N_data
        # hacky - sets [param]_indexed=[param][i] where i=k if perturbed, i=1 else }
        eval(Meta.parse(param_pert * "_indexed=" * param_pert * "[" * string(k) * "]"))
        eval(Meta.parse(param_unpert1 * "_indexed=" * param_unpert1 * "[1]"))
        eval(Meta.parse(param_unpert2 * "_indexed=" * param_unpert2 * "[1]"))
        trajectory=generate_trajectory(T,
                                       dt,
                                       state0,
                                       rho_indexed,
                                       sigma_indexed,
                                       beta_indexed)
        #trajectory = generate_trajectory(T, dt, state0, gen_traj_args)
        all_trajectories[:, :, k] = trajectory
     end

    return all_trajectories 
end

# This function is the exact same as the above, except that it takes in 
# individual inputs instead of a structure containing all of the inputs. 
# Not strictly necessary but nice because it offers us a choice of how we 
# want to generate our dataset 
function generate_dataset(N_data, T, dt, state0, rho, sigma, beta)
    all_trajectories = zeros(3, T, N_data)

    # determine which parameter is preserved by checking which matrix length exceeds 1
    param_strs = ["rho";"sigma";"beta"]
    is_perturbed = ([length(rho);length(sigma);length(beta)] .> (1,1,1)) 
    param_pert = param_strs[is_perturbed][1] 
    param_unpert1 = param_strs[.!is_perturbed][1]
    param_unpert2 = param_strs[.!is_perturbed][2]

    for k = 1:N_data
        # hacky - sets [param]_indexed=[param][i] where i=k if perturbed, i=1 else }
        eval(Meta.parse(param_pert * "_indexed=" * param_pert * "[" * string(k) * "]"))
        eval(Meta.parse(param_unpert1 * "_indexed=" * param_unpert1 * "[1]"))
        eval(Meta.parse(param_unpert2 * "_indexed=" * param_unpert2 * "[1]"))
        trajectory=generate_trajectory(T,
                                       dt,
                                       state0,
                                       rho_indexed,
                                       sigma_indexed,
                                       beta_indexed)
        #trajectory = generate_trajectory(T, dt, state0, gen_traj_args)
        all_trajectories[:, :, k] = trajectory
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

    return train_data, test_data , validation_data
end

# This is where we actually begin to build a neural net. This function is for if we want to run a 
# 2-layer NN, the first layer and second layer are both set to have 1000 nodes. Can easily 
# be modified to contain more layers, different numbers of nodes, etc.
function two_layer_model(trajectory_size; param_out=1)
    return Chain(
 	        Dense(prod(trajectory_size), 1000, relu),
            Dense(1000, param_out)
            )
end

# Same as above except now we're using a single input layer and output layer model. 
# this is just linear regression as we're using the identity operator to take us from 
# input to output. For use with our ridged regression model 
function ridge_regression_model(trajectory_size; param_out=1)
    return Dense(prod(trajectory_size), param_out, identity)
end

# This function computes the loss and accuracy of predicted parameters
# computed by our model. The loss is just the mean-squared error, and 
# accuracy is the relative error 
# Input: 
#       data_loader - an object for use with Flux, contains pairs (x, y) where x is a data point and 
#                     y is the associated parameter
#       model - this is again a Flux object, dependent on which type of neural network we want to run.
#               the two options available at the time of writing this are (1) a two-layer model and (2)
#               a ridged regression model 
#       device - another option for Flux, if gpu is available can set to run on gpu, otherwise cpu 
# Output:
#       loss / num - the mean squared loss between predicted parameters and true parameters
#       ŷ_vec - all of the predicted parameters
#       acc / num - accuracy of our prediction (the relative error between predicted and true parameter values)
function loss_and_accuracy(data_loader, model, device)
    acc = 0
    loss = 0.0f0
    num = 0
    ŷ_vec = Matrix{Float64}(undef, 1,0)
    for (x, y) in data_loader
        y = reshape(y, 1, length(y))
        x, y = device(x), device(y)
        ŷ = model(x)
        loss += mse(ŷ, y, agg=sum)
        num +=  size(x)[end]
        acc += norm(ŷ - y)/norm(y)
        ŷ_vec=[ŷ_vec ŷ]
    end
    return loss / num, ŷ_vec, acc / num
end

# same as above except now computing the ridged regression loss via penalizing the 
# weight operator via a parameter lambda, rather than the mean-squared error 
# Input: 
#       data_loader - the type of object that Flux stores the pairs (data, parameter) in
#       model -  which of the two model options to run, i.e. NN or ridge regression
#       device - this is an option that Flux has, if the computer we're running on 
#                has a gpu capable of being used for computation, can try and specify
#                here to run on the gpu, otherwise put cpu (we haven't tested this 
#                code on a gpu yet)
#       lambda - the hyperparameter used when computing the loss function
# Output:
#       loss - MSE loss with the added lambda * weights^2 for ridge regression
#       ŷ_vec - the predicted parameter values
#       acc - the accuracy of the predicted values, i.e. relative error
#       squared_error - MSE of predicted versus true
#       average - average parameter value
function ridge_regression_loss(data_loader, model, device; lambda = 0.1)
    acc = 0
    average = 0
    loss = 0.0f0
    num = 0
    squared_error = 0.0
    ŷ_vec = Matrix{Float64}(undef, 1,0)
    for (x, y) in data_loader

        y = reshape(y, 1, length(y))
        x, y = device(x), device(y)
        ŷ = model(x)

        squared_error = mse(ŷ, y, agg=sum)
        average = average + sum(y) 

        loss += squared_error + lambda * sum(model.weight.^2)
        num +=  size(x)[end]
        acc += norm(ŷ-y)/norm(y)
        
        ŷ_vec=[ŷ_vec ŷ]

    end

    return loss / num, ŷ_vec, acc / num, squared_error, average / num 

end

function test_model(data, model, device)

    ŷ_test_vec = []
    squared_error = 0.0
    num = 0.0 
    for (x,y) in data

        y = reshape(y, 1, length(y))
        x, y = device(x), device(y)
        ŷ = model(x)
        num +=  size(x)[end]
        ŷ_test_vec = [ŷ_test_vec ŷ]
        squared_error += mse(ŷ, y, agg=sum)

    end

    return ŷ_test_vec, squared_error / num

end


function train(trajectories, params, args)

    # removing this piece broke the code even though we never run on a GPU, 
    # so just leaving it here 
    if CUDA.functional() && args.use_cuda
        @info "Training on CUDA GPU"
        CUDA.allowscalar(false)
        device = gpu
    else
        @info "Training on CPU"
        device = cpu
    end

    # Load Data
    train_data, test_data, = batch_data(trajectories, params, args)

    ## Construct model
    model = args.model(length(train_data.data[1][:,1])) |> device
    ps = Flux.params(model) ## model's trainable parameters

    loss(x,y) = mse(m(x), y)

    ## Training
    evalcb = () -> @show(loss_all(train_data, m, device))
    opt = ADAM(args.η)
		
    ŷ_vec_train=[]
    ŷ_vec_test = []

    train_acc_vec = []
    test_acc_vec = []

    for epoch in 1:args.epochs
        for (x, y) in train_data
            y = reshape(y, 1, length(y))
            x, y = device(x), device(y) ## transfer data to device
            gs = Flux.gradient(() -> mse(model(x), y), ps) ## compute gradient
            Flux.Optimise.update!(opt, ps, gs) ## update parameters
        end
        
        ## Report on train and test
        # train_loss, ŷ_vec_train, train_acc = loss_and_accuracy(train_data, model, device)
        # test_loss, ŷ_vec_test, test_acc = loss_and_accuracy(test_data, model, device)

        train_loss, ŷ_vec_train, train_acc = args.loss(train_data, model, device)
        test_loss, ŷ_vec_test, test_acc = args.loss(test_data, model, device)
        println("Epoch=$epoch")
        println("train_loss = $train_loss, train_accuracy = $train_acc")
        println("test_loss = $test_loss, test_accuracy = $test_acc")
    end

    #ŷ_vec_test, test_squared_error = test_model(test_data, model, device)

    # # @show accuracy(train_data, m)

    # # @show accuracy(test_data, m)

    return train_data, test_data, ŷ_vec_train, ŷ_vec_test, train_acc_vec, test_acc_vec #, test_squared_error

end


