using Flux, Statistics
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, logitcrossentropy, throttle, @epochs
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

function split_dataset(trajectories, params, n_train, n_validation, n_test)

    x_train = trajectories[1:n_train, 1:end]
    x_validation = trajectories[n_train+1:n_validation, 1:end]
    x_test = trajectories[n_validation+1:n_test, 1:end]

    y_train = params[1:n_train]
    y_validation = params[n_train+1:n_validation]
    y_test = params[n_validation+1:n_test]

    return x_train, y_train, x_validation, y_validation, x_test, y_test

end

function getdata(trajectories, params, n_train, n_validation, n_test)
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

    # Loading Dataset	

    x_train, y_train, x_validation, y_validation, x_test, y_test = split_dataset(trajectories, params, n_train, n_validation, n_test)

    # xtrain, ytrain = # MLDatasets.MNIST.traindata(Float32)
    # xtest, ytest =  # MLDatasets.MNIST.testdata(Float32)
	
    # # Reshape Data in order to flatten each image into a linear array
    # xtrain = Flux.flatten(x_train)
    # xtest = Flux.flatten(x_test)

    # # One-hot-encode the labels
    # ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)

    # Batching
    train_data = DataLoader((x_train, y_train), batchsize=args.batchsize, shuffle=true)
    test_data = DataLoader((x_test, y_test), batchsize=args.batchsize)
    validation_data = DataLoader((x_validation, y_validation), batchsize=args.batchsize)

    return train_data, test_data, validation_data
end

function build_model(; trajectory_size=100, param_out=1)
    return Chain(
 	    Dense(prod(trajectory_size), 32, relu),
            Dense(32, param_out))
end

# function loss_all(dataloader, model)
#     L = 0f0
#     for (x,y) in dataloader
#         L += logitcrossentropy(model(x), y)
#     end
#     L/length(dataloader)
# end

function loss_all(dataloader, model)

    L = 0f0

    for (x,y) in dataloader
        L += mse(model(x), y)
    end

end

# function accuracy(data_loader, model)
#     acc = 0
#     for (x,y) in data_loader
#         acc += sum(onecold(cpu(model(x))) .== onecold(cpu(y)))*1 / size(x,2)
#     end
#     acc/length(data_loader)
# end

function train(trajectories, params, Args)
    # # Initializing model parameters 
    # args = Args(1000, 2000, 2000, 3e-4, 200, 10, gpu)

    # Load Data
    train_data,test_data = getdata(trajectories, params, Args.n_train, Args.n_validation, Args.n_test)

    # Construct model
    m = build_model()
    train_data = Args.device.(train_data)
    test_data = Args.device.(test_data)
    m = Args.device(m)
    loss(x,y) = mse(m(x), y)
    
    ## Training
    evalcb = () -> @show(loss_all(train_data, m))
    opt = ADAM(Args.η)
		
    @epochs args.epochs Flux.train!(loss, params(m), train_data, opt, cb = evalcb)

    # @show accuracy(train_data, m)

    # @show accuracy(test_data, m)

end