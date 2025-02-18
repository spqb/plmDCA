from pathlib import Path
import argparse
import numpy as np
import math
import torch

from adabmDCA.dataset import DatasetDCA
from adabmDCA.stats import get_freq_single_point, get_freq_two_points, get_correlation_two_points
from adabmDCA.utils import get_device, get_dtype
from adabmDCA.functional import one_hot

from plmDCA import plmDCA
from plmDCA.parser import add_args_train


# import command-line input arguments
def create_parser():
    # Important arguments
    parser = argparse.ArgumentParser(description='Train a plmDCA model.')
    parser = add_args_train(parser)
    
    return parser


def main():
    
    # Load parser, training dataset and DCA model
    parser = create_parser()
    args = parser.parse_args()
    
    print("\n" + "".join(["*"] * 10) + f" Training plmDCA model " + "".join(["*"] * 10) + "\n")
    # Set the device
    device = get_device(args.device)
    dtype = get_dtype(args.dtype)
    template = "{0:<30} {1:<50}"
    print(template.format("Input MSA:", str(args.data)))
    print(template.format("Output folder:", str(args.output)))
    print(template.format("Alphabet:", args.alphabet))
    print(template.format("Learning rate:", args.lr))
    print(template.format("Reciprocal couplings:", "True" if args.reciprocal else "False"))
    print(template.format("L2 reg. for fields:", args.reg_h))
    print(template.format("L2 reg. for couplings:", args.reg_J))
    print(template.format("Convergence threshold:", args.epsconv))
    if args.pseudocount is not None:
        print(template.format("Pseudocount:", args.pseudocount))
    print(template.format("Random seed:", args.seed))
    print(template.format("Data type:", args.dtype))
    print("\n")
    
    # Check if the data file exist
    if not Path(args.data).exists():
        raise FileNotFoundError(f"Data file {args.data} not found.")
    
    # Create the folder where to save the model
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    
    if args.label is not None:
        file_paths = {
            "params" : folder / Path(f"{args.label}_params.pth"),
        }
        
    else:
        file_paths = {
            "params" : folder / Path(f"params.pth"),
        }
    
    # Import dataset
    print("Importing dataset...")
    dataset = DatasetDCA(
        path_data=args.data,
        path_weights=args.weights,
        alphabet=args.alphabet,
        clustering_th=args.clustering_seqid,
        no_reweighting=args.no_reweighting,
        device=device,
        dtype=dtype,
    )
    
    # Compute statistics of the data
    L = dataset.get_num_residues()
    q = dataset.get_num_states()
    
    model = plmDCA(L=L, q=q).to(device=device, dtype=dtype)
    
    # Save the weights if not already provided
    if args.weights is None:
        if args.label is not None:
            path_weights = folder / f"{args.label}_weights.dat"
        else:
            path_weights = folder / "weights.dat"
        np.savetxt(path_weights, dataset.weights.cpu().numpy())
        print(f"Weights saved in {path_weights}")
        
    # Set the random seed
    torch.manual_seed(args.seed)
    
    if args.pseudocount is None:
        args.pseudocount = 1. / dataset.get_effective_size()
        print(f"Pseudocount automatically set to {args.pseudocount}.")
        
    data_oh = one_hot(dataset.data, num_classes=q).to(dtype)
    fi_target = get_freq_single_point(data=data_oh, weights=dataset.weights, pseudo_count=args.pseudocount)
    fij_target = get_freq_two_points(data=data_oh, weights=dataset.weights, pseudo_count=args.pseudocount)
    
    # Define the optimizer
    lr = args.lr * math.log(len(dataset)) / L # using the scaling of the learning rate proposed by GREMLIN
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print("\n")
    
    model.fit(
        X=data_oh,
        weights=dataset.weights,
        optimizer=optimizer,
        max_epochs=args.nepochs,
        epsconv=args.epsconv,
        pseudo_count=args.pseudocount,
        reg_h=args.reg_h,
        reg_J=args.reg_J,
    )
    
    if args.reciprocal:
        print("Symmetrizing the couplings...")
        model.J.data = 0.5 * (model.J.data + model.J.data.permute(2, 3, 0, 1))
    
    # Sample from the model
    # replicate the first sequence n_samples times
    print("Sampling from the model using the first sequence as wild type...")
    print("Generating 10000 sequences by doing 100 sweeps of Gibbs sampling...")
    X_init = data_oh[0].repeat(10000, 1, 1)
    nsweeps = 100
    samples = model.sample(X_init, nsweeps=nsweeps)
    pi = get_freq_single_point(samples)
    pij = get_freq_two_points(samples)
    pearson, _ = get_correlation_two_points(fi=fi_target, fij=fij_target, pi=pi, pij=pij)
    print(f"Pearson correlation of the two-site statistics: {pearson:.3f}")
    
    # Save the model
    print("Saving the model...")
    torch.save(model.state_dict(), file_paths["params"])
    print(f"Model saved in {file_paths['params']}")
    
    
if __name__ == "__main__":
    main()