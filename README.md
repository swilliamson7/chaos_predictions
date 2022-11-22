### This is a repository for an ongoing project by Matt Goldberg and Sarah Williamson, graduate students at the University of Texas at Austin. 

# Parameter estimation in a chaotic system 

We'd like to try and investigate how the adjoint method compares to a neural network when it comes to parameter estimation. Some of the questions we want to answer/work on are  

1. At what time scales does the adjoint method cease to be an effective method for parameter estimation? 
2. Could we potentially use a combination of ML and the adjoint to better predict an uncertain parameter?
3. Can we quantify, in a rigorous mathematical sense, what ML does to predict a parameter that the adjoint method cannot?

Note -- when we say we want to use the adjoint method for parameter estimation what we're hiding is a full description: really we (1) use the adjoint method to compute a gradient of our system w.r.t. the uncertain parameter (2) use that gradient with an optimization method (in our case we'll be using gradient descent and seeing what results this gives) and then (3) hopefully converge to a closer parameter. 

# Overview of repository contents

We aim to consider these questions for a few different systems. The code for each system is in it's corresponding folder (i.e., Lorenz code is in Lorenz). The systems we have considered thus far are (1) a one-dimensional forced pendulum and (2) the Lorenz model. We aim to move to a true ocean model (using Ocenanigans) eventually.

Each folder contains a number of scripts. In general, all functions needed to run our experiments are contained in scripts labeled 

    1. create_structs.jl
    
    2. [which model]_model.jl
    
    3. neural_net_[which model].jl
    
    4. create_structs.jl
    
    5. plotting.jl
    
where [which model] means to fill in with the desired one. Then experiments are run in the Julia scripts with "experiment" in the name. In general, we focused more on the lorenz model thus far. If one wants to try and run our code for the Lorenz model, one only needs to run say, parameter"underscore"experiment.jl. Download the folder titled "Lorenz," navigate to it in the terminal (or open in the directory in VSCode or your favorite editor), and then run the line 

```julia
include("parameter_experiment.jl"
```

which will include all of the scripts needed to run the experiment and output some resulting plots showing test parameters versus the predicted values. 

# Lorenz model 

We ran two types of experiments with the Lorenz model thus far. The first was simple: we generated multiple trajectories of the Lorenz model with slightly varying $\sigma$ values, gave as input to a ridge regression model sparse noisy observations from these timeseries, and attempted to predict what $\sigma$ value was used to create them. The second experiment was to see what happened if we used the output of the 

# Barotropic gyre model 

-----------------

00. Read the papers Patrick sent

0. Maybe try setting up the chaotic pendulum problem the way that they did in the paper, just as a learning tool. 

    (a) could help to understand what controllability means in this context. 

1. Setting up code that runs our "truth" model, this will be used to create the data that we want. 

2. Adjointing the above code. It will compute a gradient but it won't compute a useful gradient.

3. Thinking about how we can use "controllability" to minimize cost function using the gradient that was found in part 2. This will entail 

    (a) thinking about how the pendulum paper accomplished their task. 
        
    (b) thinking about how to set up the same experiment with the Lorenz model, especially considering that we don't have forcing. 
        
    (c) how to set it up for parameter estimation, and not forcing estimation. 
        
 4. We can try (3) with different methods/models, maybe Lorenz 86 and 96? 
 
 5. The last thing we want to try is to train a neural network on the results of a data assimilation scheme, similar to what was done in yet more papers.
 
    (a) set up neural network to learn from the results of a DA scheme 
    
    (b) learn machine learning
