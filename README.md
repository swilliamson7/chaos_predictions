# chaos_predictions

Attempting to run both a modified adjoint method as well as a combination of machine learning and data assimilation. The steps that we can try to complete are:

0. Try setting up the chaotic pendulum problem the way that they did in the paper, just as a learning tool. 

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
