# This script will keep track of my first attempt at recreating the code done in the paper by 
# Geoffrey Gebbie and Tsung-Lin Hsieh. 

using LinearAlgebra, Plots, Enzyme, Optim 

# this contains a bunch of functions that will be used throughout this code  
include("pend_model.jl")

const T = 50            # how long we'll run the forced pendulum for (seconds)
const dt = .01          # distance between steps
const nt = Int(T / dt)  # number of steps to take, given the above T and dt 

