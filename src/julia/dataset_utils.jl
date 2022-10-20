using JLD2 

function load_dataset(out_dir_dataset_filename, generate_dataset_args)
    if isfile(out_dir_dataset_filename)
        dataset=load_object(out_dir_dataset_filename)
    else
        dataset=generate_dataset(generate_dataset_args)
        save_object(out_dir_dataset_filename, dataset)
    end
    return dataset
end