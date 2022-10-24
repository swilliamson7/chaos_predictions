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

function generate_dataset(s::generate_dataset_Args)
    @unpack_generate_dataset_Args s
    theta = zeros(N_data, T+1)

    for k = 1:N_data

        trajectory = generate_trajectory(T, dt, b, g, state0, q[k], l)
        theta[k, :] = trajectory[2, :]
        
    end

    return theta 
end

function generate_dataset(N_data, T, dt, b, g, state0, q, l)

    theta = zeros(N_data, T+1)

    for k = 1:N_data

        trajectory = generate_trajectory(T, dt, b, g, state0, q[k], l)
        theta[k, :] = trajectory[2, :]
        
    end

    return theta 
end

function generate_dataset_perturb_b(s::generate_dataset_Args)
    @unpack_generate_dataset_Args s
    theta = zeros(N_data, T+1)

    for k = 1:N_data

        trajectory = generate_trajectory(T, dt, b[k], g, state0, q[1], l[1])
        theta[k, :] = trajectory[2, :]
        
    end

    return theta 
end

function split_dataset(trajectories, params, Args)

    x_train = trajectories[1:Args.n_train, 1:end]
    x_test = trajectories[Args.n_train+1:Args.n_test+Args.n_train, 1:end]
    x_validation = trajectories[Args.n_train+Args.n_test+1:end, 1:end]

    y_train = params[1:Args.n_train]
    y_test = params[Args.n_train+1:Args.n_test+Args.n_train]
    y_validation = params[Args.n_train + Args.n_test + 1:end]

    return x_train, y_train, x_test, y_test, x_validation, y_validation

end


function getdata(trajectories, params, Args)
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

    # Loading Dataset	

    x_train, y_train, x_test, y_test, x_validation, y_validation = split_dataset(trajectories, params, Args)

    # xtrain, ytrain = # MLDatasets.MNIST.traindata(Float32)
    # xtest, ytest =  # MLDatasets.MNIST.testdata(Float32)
	
    # Reshape Data in order to flatten each image into a linear array
    x_train = x_train'
    x_test = x_test'

    # # One-hot-encode the labels
    # ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)

    # Batching
    train_data = DataLoader((x_train, y_train), batchsize=Args.batchsize, shuffle=true)
    test_data = DataLoader((x_test, y_test), batchsize=Args.batchsize)
    #validation_data = DataLoader((x_validation, y_validation), batchsize=Args.batchsize)

    return train_data, test_data #, validation_data
end

function build_model(; trajectory_size=501, param_out=1)
    return Chain(
 	    Dense(prod(trajectory_size), 1000, relu),
 	    Dense(1000, 500, relu),
        Dense(500, param_out))
end

#function build_model(; trajectory_size=501, param_out=1)
#    return Chain(
# 	    Dense(prod(trajectory_size), 1000, relu),
#        Dense(1000, param_out))
#end


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

function train(trajectories, params, args)
    # Initializing model parameters 
    

    if CUDA.functional() && args.use_cuda
        @info "Training on CUDA GPU"
        CUDA.allowscalar(false)
        device = gpu
    else
        @info "Training on CPU"
        device = cpu
    end

    # Load Data
    train_data, test_data = getdata(trajectories, params, args)

    ## Construct model
    model = build_model() |> device
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
        train_loss, ŷ_vec_train, train_acc = loss_and_accuracy(train_data, model, device)
        test_loss, ŷ_vec_test, test_acc = loss_and_accuracy(test_data, model, device)
        println("Epoch=$epoch")
        println("  train_loss = $train_loss, train_accuracy = $train_acc")
        println("  test_loss = $test_loss test_accuracy = $test_acc")
        train_acc_vec=append!(train_acc_vec, train_acc)
        test_acc_vec=append!(test_acc_vec, test_acc)
    end

    return train_data, test_data, ŷ_vec_train, ŷ_vec_test, train_acc_vec, test_acc_vec

end


