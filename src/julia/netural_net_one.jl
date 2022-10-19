using Flux, Statistics
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, logitcrossentropy, throttle, @epochs
using Flux.Losses: mse
using Base.Iterators: repeated
using Parameters: @with_kw
using CUDA
using MLDatasets
using JLD2 

include("pend_model.jl")

@load "q_values.jld2" q_values 
@load "l_values.jld2" l_values 

if has_cuda()		# Check if CUDA is available
    @info "CUDA is on"
    CUDA.allowscalar(false)
end

# @with_kw mutable struct Args
#     η::Float64 = 3e-4       # learning rate
#     batchsize::Int = 200    # batch size
#     epochs::Int = 10        # number of epochs
#     device::Function = gpu  # set as gpu, if gpu available
# end

function generate_dataset(N_data, T, dt, b, g, state0, q, l)

    theta = zeros(N_data, T+1)

    for k = 1:N_data

        trajectory = generate_trajectory(T, dt, k, b, g, state0, q[k], l)
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

function build_model(; trajectory_size=101, param_out=1)
    return Chain(
 	    Dense(prod(trajectory_size), 1000, relu),
            Dense(1000, param_out))
end

# function loss_all(dataloader, model)
#     L = 0f0
#     for (x,y) in dataloader
#         L += logitcrossentropy(model(x), y)
#     end
#     L/length(dataloader)
# end

# function loss_all(dataloader, model)

#     L = 0f0

#     for (x,y) in dataloader
#         L += mse(model(x), y)
#     end

# end

function loss_all(data_loader, model, device)
    #acc = 0
    ls = 0.0f0
    num = 0
    for (x, y) in data_loader
        y = reshape(y, 1, length(y))
        x, y = device(x), device(y)
        ŷ = model(x)
        ls += mse(ŷ, y, agg=sum)
        #acc += sum(onecold(ŷ) .== onecold(y)) ## Decode the output of the model
        num +=  size(x)[end]
    end
    return ls / num #acc / num
end

# function accuracy(data_loader, model)
#     acc = 0
#     for (x,y) in data_loader
#         acc += sum(onecold(cpu(model(x))) .== onecold(cpu(y)))*1 / size(x,2)
#     end
#     acc/length(data_loader)
# end

function train(trajectories, params, Args)
    # Initializing model parameters 
    args = Args(4000, 5000, 4000, 3e-4, 200, 500, gpu)

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

    loss(x,y) = mse(m(x), y)

    ## Training
    evalcb = () -> @show(loss_all(train_data, m, device))
    opt = ADAM(args.η)
		
    for epoch in 1:args.epochs
        for (x, y) in train_data
            y = reshape(y, 1, length(y))
            x, y = device(x), device(y) ## transfer data to device
            gs = gradient(() -> mse(model(x), y), ps) ## compute gradient
            Flux.Optimise.update!(opt, ps, gs) ## update parameters
        end
        
        ## Report on train and test
        train_loss = loss_all(train_data, model, device)
        test_loss = loss_all(test_data, model, device)
        println("Epoch=$epoch")
        println("  train_loss = $train_loss")#, train_accuracy = $train_acc")
        println("  test_loss = $test_loss")#, test_accuracy = $test_acc")
    end

    # # @show accuracy(train_data, m)

    # # @show accuracy(test_data, m)

    return train_data, test_data

end