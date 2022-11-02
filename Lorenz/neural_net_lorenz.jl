using Flux, Statistics
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, logitcrossentropy, throttle, @epochs
using Flux.Losses: mse
using Base.Iterators: repeated
using Parameters: @with_kw
using CUDA
using MLDatasets
using JLD2 
using Debugger
using LinearAlgebra

include("create_structs.jl")
include("lorenz_model.jl")

if has_cuda()		# Check if CUDA is available
    @info "CUDA is on"
    CUDA.allowscalar(false)
end

# generates N_data different trajectories to use as our data points 
function generate_dataset(😄::generate_dataset_Args)
    @unpack_generate_dataset_Args 😄
    all_trajectories = zeros(3, T, N_data)

    if length(rho) > 1
        for k = 1:N_data
            trajectory = generate_trajectory(T, dt, state0, rho[k], sigma[1], beta[1])
            all_trajectories[:, :, k] = trajectory
        end
    end
    if length(sigma) > 1 
        for k = 1:N_data
            trajectory = generate_trajectory(T, dt, state0, rho[1], sigma[k], beta[1])
            all_trajectories[:, :, k] = trajectory
        end
    end
    if length(beta) > 1 
        for k = 1:N_data
            trajectory = generate_trajectory(T, dt, state0, rho[1], sigma[1], beta[k])
            all_trajectories[:, :, k] = trajectory
        end
    end

    return all_trajectories 

end

function generate_dataset(N_data, T, dt, state0, rho, sigma, beta)

    all_trajectories = zeros(3, T, N_data)

    if length(rho) > 1
        for k = 1:N_data
            trajectory = generate_trajectory(T, dt, state0, rho[k], sigma[1], beta[1])
            all_trajectories[:, :, k] = trajectory[:, :]
        end
    elseif length(sigma) > 1 
        for k = 1:N_data
            trajectory = generate_trajectory(T, dt, state0, rho[1], sigma[k], beta[1])
            all_trajectories[:, :, k] = trajectory[:, :]
        end
    elseif length(beta) > 1 
        for k = 1:N_data
            trajectory = generate_trajectory(T, dt, state0, rho[1], sigma[1], beta[k])
            all_trajectories[:, :, k] = trajectory[:, :]
        end
    end

    return all_trajectories 

end

# # Takes all of the trajectories and transforms into a matrix, batched as 
# # train, validation, and test. Specifically, the output of Flux.flatten(data)
# # is a matrix where each column is one data point (i.e. three-dimensional trajectory) 
# function flatten_trajectories(trajectories, Args::train_Args)

#     train_trajectories = Flux.flatten(trajectories[:, :, 1:Args.n_train])

#     validation_trajectories = Flux.flatten(trajectories[:, :, Args.n_train + 1:Args.n_validation + Args.n_train])

#     test_trajectories = Flux.flatten(trajectories[:, :, Args.n_validation + Args.n_train + 1:end])

#     return train_trajectories, validation_trajectories, test_trajectories
    
# end

# After generating a bunch of different trajectories we split them into the 
# canonical train, validate, test groups.
function split_dataset(trajectories, params, Args::train_Args)

    x_train = trajectories[1:Args.n_train, :]
    x_test = trajectories[Args.n_train+1:Args.n_test+Args.n_train, :]
    x_validation = trajectories[Args.n_train+Args.n_test+1:end, :]

    y_train = params[1:Args.n_train]
    y_test = params[Args.n_train+1:Args.n_test+Args.n_train]
    y_validation = params[Args.n_train + Args.n_test + 1:end]

    return x_train, y_train, x_test, y_test, x_validation, y_validation

end

# Separates the given data points into batches corresponding to train, 
# validation, and test
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

function build_model(trajectory_size; param_out=1)
    return Chain(
 	    Dense(prod(trajectory_size), 1000, relu),
            Dense(1000, param_out))
end

# same as above except now we're using a single input layer and output layer model. 
# this is just linear regression as we're using the identity operator to take us from 
# input to output 
function regression_model(trajectory_size; param_out=1)
    return Dense(prod(trajectory_size), param_out, identity)
end

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
# weight operator via a parameter lambda 
function ridge_regression_loss(data_loader, model, device, lambda)
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

    # Load Data
    train_data, test_data, = batch_data(trajectories, params, args)

    
    if CUDA.functional() && args.use_cuda
        @info "Training on CUDA GPU"
        CUDA.allowscalar(false)
        device = gpu
    else
        @info "Training on CPU"
        device = cpu
    end

    # Load Data
    train_data, test_data, _ = batch_data(trajectories, params, args)

    ## Construct model
    model = regression_model(length(train_data.data[1][:,1])) |> device
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
            gs = gradient(() -> mse(model(x), y), ps) ## compute gradient
            Flux.Optimise.update!(opt, ps, gs) ## update parameters
        end
        
        ## Report on train and test
        # train_loss, ŷ_vec_train, train_acc = loss_and_accuracy(train_data, model, device)
        # test_loss, ŷ_vec_test, test_acc = loss_and_accuracy(test_data, model, device)

        train_loss, ŷ_vec_train, train_acc = ridge_regression_loss(train_data, model, device, 0.1)
        test_loss, ŷ_vec_test, test_acc = ridge_regression_loss(test_data, model, device, 0.1)
        println("Epoch=$epoch")
        println("train_loss = $train_loss, train_accuracy = $train_acc")
        println("test_loss = $test_loss, test_accuracy = $test_acc")
    end

    #ŷ_vec_test, test_squared_error = test_model(test_data, model, device)

    # # @show accuracy(train_data, m)

    # # @show accuracy(test_data, m)

    return train_data, test_data, ŷ_vec_train, ŷ_vec_test, train_acc_vec, test_acc_vec #, test_squared_error

end


