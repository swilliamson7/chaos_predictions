# Contains all the functions that we use in our neural net/linear regression model 

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
include("pend_model.jl")

@load "q_values.jld2" q_values 
@load "l_values.jld2" l_values 

if has_cuda()		# Check if CUDA is available
    @info "CUDA is on"
    CUDA.allowscalar(false)
end


# generates N_data different trajectories to use as our data points 
function generate_dataset(s::generate_dataset_Args)
    @unpack_generate_dataset_Args s
    theta = zeros(N_data, T+1)

    if length(b) > 1
        for k = 1:N_data
            trajectory = generate_trajectory(T, dt, b[k], g, state0, q[1], l[1])
            theta[k, :] = trajectory[2, :]
        end
    end
    if length(q) > 1 
        for k = 1:N_data
            trajectory = generate_trajectory(T, dt, b[1], g, state0, q[k], l[1])
            theta[k, :] = trajectory[2, :]
        end
    end
    if length(l) > 1 
        for k = 1:N_data
            trajectory = generate_trajectory(T, dt, b[1], g, state0, q[1], l[k])
            theta[k, :] = trajectory[2, :]
        end
    end

    return theta 

end

# same as above, but doesn't need a structure as input, instead takes the individual inputs 
function generate_dataset(N_data, T, dt, b, g, state0, q, l)

    theta = zeros(N_data, T+1)

    if length(b) > 1
        for k = 1:N_data
            trajectory = generate_trajectory(T, dt, b[k], g, state0, q[1], l[1])
            theta[k, :] = trajectory[2, :]
        end
    elseif length(q) > 1 
        for k = 1:N_data
            trajectory = generate_trajectory(T, dt, b[1], g, state0, q[k], l[1])
            theta[k, :] = trajectory[2, :]
        end
    elseif length(l) > 1 
        for k = 1:N_data
            trajectory = generate_trajectory(T, dt, b[1], g, state0, q[1], l[k])
            theta[k, :] = trajectory[2, :]
        end
    end

    return theta 

end

# separates the N_data different trajectories into train, test, and validation buckets,
# the size of these buckets is kept in Args  
function split_dataset(trajectories, params, Args)

    x_train = trajectories[1:Args.n_train, 1:end]
    x_test = trajectories[Args.n_train+1:Args.n_test+Args.n_train, 1:end]
    x_validation = trajectories[Args.n_train+Args.n_test+1:end, 1:end]

    y_train = params[1:Args.n_train]
    y_test = params[Args.n_train+1:Args.n_test+Args.n_train]
    y_validation = params[Args.n_train + Args.n_test + 1:end]

    return x_train, y_train, x_test, y_test, x_validation, y_validation

end

# this creates data for Flux to use, seemingly pairs a trajectory with the parameter
# value that generated it and returns these in a data structure 
function getdata(trajectories, params, Args)
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

    # Loading Dataset	

    x_train, y_train, x_test, y_test, x_validation, y_validation = split_dataset(trajectories, params, Args)
	
    # Reshape Data in order to flatten each image into a linear array
    x_train = x_train'
    x_test = x_test'
    x_validation = x_validation'

    # Batching
    train_data = DataLoader((x_train, y_train), batchsize=Args.batchsize, shuffle=true)
    test_data = DataLoader((x_test, y_test), batchsize=Args.batchsize)
    validation_data = DataLoader((x_validation, y_validation), batchsize=Args.batchsize)

    return train_data, test_data, validation_data
end


# this is designed to be the layers in the NN, each call of Dense is one more layer 
function build_model(; trajectory_size=1001, param_out=1)
    return Chain(
 	    Dense(prod(trajectory_size), 1000, relu),
 	    Dense(1000, 500, relu),
        Dense(500, param_out))
end

# same as above except now we're using a single input layer and output layer model. 
# this is just linear regression as we're using the identity operator to take us from 
# input to output 
function regression_model(; trajectory_size=1001, param_out=1)
    return Dense(prod(trajectory_size), param_out, identity)
end

# computes the loss and accuracy on the desired model 
function loss_and_accuracy(data_loader, model, device)
    acc = 0
    ls = 0.0f0
    num = 0
    ŷ_vec = Matrix{Float64}(undef, 1,0)
    for (x, y) in data_loader
        y = reshape(y, 1, length(y))
        x, y = device(x), device(y)
        ŷ = model(x)
        ls += mse(ŷ, y, agg=sum)
        num +=  size(x)[end]
        acc += norm(ŷ-y)/norm(y)
        ŷ_vec=[ŷ_vec ŷ]
    end
    return ls / num, ŷ_vec, acc / num
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

# does the training based on what was built above 
function train(trajectories, params, args, lambda)

    # Load Data
    train_data, test_data = getdata(trajectories, params, args)

    if CUDA.functional() && args.use_cuda
        @info "Training on CUDA GPU"
        CUDA.allowscalar(false)
        device = gpu
    else
        @info "Training on CPU"
        device = cpu
    end

    ## Construct model
    # model = build_model() |> device
    model = regression_model() |> device 
    ps = Flux.params(model) ## model's trainable parameters

    ## Optimizer
    opt = ADAM(args.η)
		
    ## Training
    ŷ_vec_train=[]
    ŷ_vec_test=[]
    train_acc_vec=[]
    test_acc_vec=[]
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

        train_loss, ŷ_vec_train, train_acc = ridge_regression_loss(train_data, model, device, lambda)
        test_loss, ŷ_vec_test, test_acc = ridge_regression_loss(test_data, model, device, lambda)

        println("Epoch=$epoch")
        println("train_loss = $train_loss, train_accuracy = $train_acc")
        println("test_loss = $test_loss test_accuracy = $test_acc")
        train_acc_vec=append!(train_acc_vec, train_acc)
        test_acc_vec=append!(test_acc_vec, test_acc)
    end

   return train_data, test_data, ŷ_vec_train, ŷ_vec_test, train_acc_vec, test_acc_vec

end

# this function will compute the losses we see for various lambda values 
function hyperparameter_training(trajectories, params, args, lambdas)

    # Load Data
    train_data, test_data, validation_data = getdata(trajectories, params, args)

    if CUDA.functional() && args.use_cuda
        @info "Training on CUDA GPU"
        CUDA.allowscalar(false)
        device = gpu
    else
        @info "Training on CPU"
        device = cpu
    end

    ## Construct model
    model = regression_model() |> device 
    ps = Flux.params(model) ## model's trainable parameters

    ## Optimizer
    opt = ADAM(args.η)

    validation_losses = []
    ŷ_vec_validation=[]
    r_squared_vec = []
	
    for lambda in lambdas 
        ## Training
        ŷ_vec_train=[]
        train_acc_vec=[]
        test_acc_vec=[]
        for epoch in 1:args.epochs
            for (x, y) in train_data
                y = reshape(y, 1, length(y))
                x, y = device(x), device(y) ## transfer data to device
                gs = gradient(() -> mse(model(x), y), ps) ## compute gradient
                Flux.Optimise.update!(opt, ps, gs) ## update parameters
            end
            
            train_loss, ŷ_vec_train, train_acc, squared_error_train, average_train  = ridge_regression_loss(train_data, model, device, lambda)

            # println("Epoch=$epoch")
            # println("train_loss = $train_loss, train_accuracy = $train_acc")
            # println("validation_loss = $test_loss validation_accuracy = $test_acc")
            train_acc_vec=append!(train_acc_vec, train_acc)
        end
        validation_loss, ŷ_vec_validation, validation_acc, squared_error_validation, average_validation = ridge_regression_loss(validation_data, model, device, lambda)
        push!(validation_losses, validation_loss)
        push!(r_squared_vec, r_squared(squared_error_validation, sum((validation_data.data[2] .- average_validation).^2)))

    end

   return ŷ_vec_validation, validation_losses, r_squared_vec

end 

function r_squared(u, v)
    return 1 - (u / v)
end