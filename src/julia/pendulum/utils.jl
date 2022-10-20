

function normalize_range(values, out_range)
    old_min, old_max = [minimum(values), maximum(values)]
    new_max, new_min = out_range
    normalized_values = (((values .- old_min) .* (new_max .- new_min)) ./ (old_max .- old_min)) .+ new_min
    return normalized_values
end


params = 50 .+ 50 .* randn(1, 100)
params_normalized = normalize_range(params, [99.0, 101.0])
