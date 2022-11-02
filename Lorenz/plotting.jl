using Plots

function plot_data_pred_vs_true(every_nth, pred_vec, true_vec; pred_label="ŷ_vec_train", true_label="y_vec_train", kwargs...)
    x=1:length(pred_vec)
    plot(x[1:every_nth:end], pred_vec[1:every_nth:end], seriestype = :scatter, label = pred_label; kwargs...) 
    plot!(x[1:every_nth:end], true_vec[1:every_nth:end], seriestype = :scatter, label = true_label; kwargs...) 
end


# plot accuracy
function plot_acc(epochs, acc_vec, label)
    return plot(1:epochs, acc_vec, seriestype = :scatter, label = label, xlabel="epoch", ylabel=L"\frac{||\theta_{pred}-\theta_{true}||}{||\theta_{pred}||}", leftmargin=8mm) 
end

function plot_data_pred_minus_true(every_nth, pred_vec, true_vec; kwargs...)
    x=1:length(pred_vec)
    plot(x[1:every_nth:end], pred_vec[1:every_nth:end]-true_vec[1:every_nth:end], seriestype = :scatter, kwargs...) 
end