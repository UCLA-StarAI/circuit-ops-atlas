using LogicCircuits, ProbabilisticCircuits


function main(dataset_name, niter)
    train_data, valid_data, test_data = twenty_datasets(dataset_name)
    
    pc, vtree = learn_circuit(train_data; maxiter = niter, return_vtree = true)
    
    save_as_psdd("./pcs/$(dataset_name)_$(string(niter)).psdd", pc, vtree)
end

dataset_id = parse(Int64, ARGS[1])
dataset_name = twenty_dataset_names[dataset_id]

niter = parse(Int64, ARGS[2])

main(dataset_name, niter)