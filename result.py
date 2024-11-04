import pickle

file_path = "/home/fan.zhang/base/pl-logdir/phenomd-60-transforms-4-4-resnet-wider-dl/flow-samples-as-bilby-result_1000.pickle"

with open(file_path, 'rb') as f:
    results = pickle.load(f)

for idx, result in enumerate(results):
    print(f"Result {idx}:")
    

    print("Posterior DataFrame:")
    print(result.posterior)
    

    print("\nInjection Parameters:")
    print(result.injection_parameters)
    

    print("\nPriors:")
    print(result.priors)
  
    print("\nParameter Labels:")
    print(result.parameter_labels)
    
 
    print(f"\nNumber of samples: {result.posterior.shape[0]}") 

    print("---------\n")

