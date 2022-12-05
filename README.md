### This is a repository for an ongoing project by Matt Goldberg and Sarah Williamson, graduate students at the University of Texas at Austin. 

# Parameter estimation in a chaotic system 

We'd like to try and investigate how the adjoint method compares to a neural network when it comes to parameter estimation. Some of the questions we want to answer/work on are  

1. At what time scales does the adjoint method cease to be an effective method for parameter estimation? 
2. Could we potentially use a combination of ML and the adjoint to better predict an uncertain parameter?
3. Can we quantify, in a rigorous mathematical sense, what ML does to predict a parameter that the adjoint method cannot?

Note -- when we say we want to use the adjoint method for parameter estimation what we're hiding is a full description: really we (1) use the adjoint method to compute a gradient of our system w.r.t. the uncertain parameter (2) use that gradient with an optimization method (in our case we'll be using gradient descent and seeing what results this gives) and then (3) hopefully converge to a closer parameter. 

# Overview of repository contents

We aim to consider these questions for a few different systems. The code for each system is in it's corresponding folder (i.e., Lorenz code is in Lorenz). The systems we have considered thus far are (1) a one-dimensional forced pendulum and (2) the Lorenz model. We aim to move to a true ocean model (using Ocenanigans) eventually. The pendulum model lives in Old_Files, namely because we abandoned that as a potential toy model in favor of the Lorenz system early on. 

Lorenz contains a number of scripts. In general, all functions needed to run our experiments are contained in scripts labeled 

    1. create_structs.jl
    
    2. lorenz_model.jl
    
    3. neural_net_lorenz.jl
    
    4. create_structs.jl
    
    5. plotting.jl
    
Then experiments are run in the Julia scripts with "experiment" in the name. In general, we've mainly focused on the Lorenz model thus far. If someone wants to try and run our code for the Lorenz model, one only needs to run say, parameter"underscore"experiment.jl. Download the folder titled "Lorenz," navigate to it in the terminal (or open in the directory in VSCode or your favorite editor), and then run the line 

```julia
include("parameter_experiment.jl")
```

which will include all of the scripts needed to run the neural net experiment and output some resulting plots showing test parameters versus the predicted values. Same goes for running the adjoint, we first include the file

```julia
include("adjoint_experiment.jl")
```

which initializes all of the scripts and functions we need. Then, as an example run, 

```julia
x0 = [1.0, 0.0, 0.0]               
rho = 28.0                           
sigma = 10.0                        
sigma_guess = 10.3 
beta = 8/3 
every_nth = 2 
Ts = [100 + 10*k for k = 0:21]

sigmas = adjoint_experiment(state0, rho, sigma, sigma_guess, beta, every_nth, Ts)

plot(Ts, sigmas, seriestype = :scatter, label = "", xlabel="Integration time", ylabel=L"\sigma", dpi = 300)

```

This will return a plot of integration time versus $\sigma$. The adjoint method is currently only setup to find $\sigma$, will be updated soon. 

# Lorenz model 

We ran two types of experiments with the Lorenz model thus far. The first was simple: we generated multiple trajectories of the Lorenz model with slightly varying $\sigma$ values, gave as input to a machine learning model sparse noisy observations from these timeseries, and attempted to predict what $\sigma$ value was used to create them. The predicted $\sigma$ value was found in one of two ways, either via the adjoint method (adjoint_experiment.jl) or with machine learning. 

The adjoint method (understandbly) fails at predicting the parameter when we integrate the model for a long time. On the other hand, machine learning predicts the parameter reliably, at least under the specifics of our experiments (relatively low noise, integrated for 500 steps, etc.) There are plenty of other options to play around with to see how consistently the method performs, and we've only really scratched the surface. 

# Barotropic gyre model 

It would be interesting to set up a double gyre toy model (i.e. get some eddy behavior) and perform similar experiments. We'll need to be more clever about how we generate our trajectories though, as it takes substantially more time to run a toy ocean model.

-----------------
