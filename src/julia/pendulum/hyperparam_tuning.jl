using Hyperopt

function hyperTune()
ho = @hyperopt for i = 20,
             η=0.00005:0.00005:0.0003,
             batchsize=StepRange(50:50:400),
             @show train(trajectories, params, args)
function train(trajectories, params, args)
