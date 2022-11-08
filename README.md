### This is a repository for an ongoing project by Matt Goldberg and Sarah Williamson, graduate students at the University of Texas at Austin. 

# Parameter estimation in a chaotic system 

We'd like to try and investigate how the adjoint method compares to a neural network when it comes to parameter estimation. Some of the questions we want to answer/work on are  

1. At what time scales does the adjoint method cease to be an effective method for parameter estimation? 
2. Could we potentially use a combination of ML and the adjoint to better predict an uncertain parameter?
3. Can we quantify, in a rigorous mathematical sense, what ML does to predict a parameter that the adjoint method cannot?

Note -- when we say we want to use the adjoint method for parameter estimation what we're hiding is a full description: really we (1) use the adjoint method to compute a gradient of our system w.r.t. the uncertain parameter (2) use that gradient with an optimization method (in our case we'll be using ????) and then (3) hopefully converge to a closer parameter. 

# Overview of repository contents

We aim to consider these questions for a few different systems. The code for each system is in it's corresponding folder (i.e., Lorenz code is in Lorenz). 


Attempting to run both a modified adjoint method as well as a combination of machine learning and data assimilation. The steps that we can try to complete are:

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
