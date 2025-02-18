import argparse
from pathlib import Path
import torch
import pandas as pd

from adabmDCA.fasta import (
    get_tokens,
    write_fasta,
)
from plmDCA import plmDCA
from plmDCA.parser import add_args_sample
from adabmDCA.dataset import DatasetDCA
from adabmDCA.utils import resample_sequences, get_device, get_dtype
from adabmDCA.functional import one_hot
from adabmDCA.stats import get_freq_single_point, get_freq_two_points, get_correlation_two_points


# import command-line input arguments
def create_parser():
    # Important arguments
    parser = argparse.ArgumentParser(description='Samples from a DCA model.')
    parser = add_args_sample(parser)
    
    return parser


def main():       
    
    # Parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Create output folder
    filename = Path(args.output)
    folder = filename.parent / Path(filename.name)
    # Create the folder where to save the samples
    folder.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "".join(["*"] * 10) + f" Sampling from plmDCA model " + "".join(["*"] * 10) + "\n")
    # Set the device
    device = get_device(args.device)
    dtype = get_dtype(args.dtype)
    tokens = get_tokens(args.alphabet)
    
    # Check that the data file exists
    if not Path(args.data).exists():
        raise FileNotFoundError(f"Data file {args.data} not found.")
    
    # Check that the parameters file exists
    if not Path(args.path_params).exists():
        raise FileNotFoundError(f"Parameters file {args.path_params} not found.")    
        
    # Import parameters
    print(f"Loading parameters from {args.path_params}...")
    params = torch.load(args.path_params)
    L, q = params["h"].shape
    model = plmDCA(L=L, q=q).to(device=device, dtype=dtype)
    model.load_state_dict(params)
    
    # Load the data
    print(f"Loading data from {args.data}...")
    dataset = DatasetDCA(
        path_data=args.data,
        path_weights=args.weights,
        alphabet=tokens,
        clustering_th=args.clustering_seqid,
        no_reweighting=args.no_reweighting,
        device=device,
        dtype=dtype,
    )
    data = one_hot(dataset.data, num_classes=len(tokens)).to(dtype)
    # Compute single and two-site frequencies of the data
    if args.pseudocount is None:
        args.pseudocount = 1. / dataset.weights.sum()
    print(f"Using pseudocount: {args.pseudocount}...")
    fi = get_freq_single_point(data=data, weights=dataset.weights, pseudo_count=args.pseudocount)
    fij = get_freq_two_points(data=data, weights=dataset.weights, pseudo_count=args.pseudocount)
    # Select the initial condition
    if args.init_random:
        X = torch.randint(0, q, (args.ngen, data.shape[1], data.shape[2]), device=device, dtype=dtype)
    else:
        # Take the first sequence as the wild-type
        X = data[0].repeat(args.ngen, 1, 1)
        
    # Sample from the model
    print(f"Sampling {args.ngen} sequences...")
    r_f_i = []
    r_C_ij = []
    for sweep in range(args.nsweeps):
        # Compute the statistics of the generated sequences
        pi = get_freq_single_point(data=X)
        pij = get_freq_two_points(data=X)
        corr_one_site = torch.corrcoef(torch.stack([fi.flatten(), pi.flatten()]))[0, 1].item()
        corr_two_sites, _ = get_correlation_two_points(fi=fi, pi=pi, fij=fij, pij=pij)
        r_f_i.append(corr_one_site)
        r_C_ij.append(corr_two_sites)
        if sweep in [50, 100, 200, 500, 1000]:
            write_fasta(
                fname=folder / Path(f"{args.label}_samples_{sweep}.fasta"),
                headers=[f"sequence {i+1}" for i in range(args.ngen)],
                sequences=X.argmax(-1).cpu(),
                numeric_input=True,
                tokens=tokens,
            )
        
        # Perform one sweep
        residue_idxs = torch.randperm(L)
        X = model(X, residue_idxs, beta=args.beta)
    
    # Save the log
    log = pd.DataFrame({
        "sweep": list(range(args.nsweeps)),
        "corr_one_site": r_f_i,
        "corr_two_sites": r_C_ij,
    })
    log.to_csv(folder / Path(f"{args.label}_log.csv"), index=False)
    
    pi = get_freq_single_point(data=X)
    pij = get_freq_two_points(data=X)
    pearson, slope = get_correlation_two_points(fi=fi, pi=pi, fij=fij, pij=pij)
    print(f"Pearson correlation coefficient: {pearson:.3f}")
    print(f"Slope: {slope:.3f}")
    print(f"Saving the sequences...")
    write_fasta(
        fname=folder / Path(f"{args.label}_samples_{args.nsweeps}.fasta"),
        headers=[f"sequence {i+1}" for i in range(args.ngen)],
        sequences=X.argmax(-1).cpu(),
        numeric_input=True,
        tokens=tokens,
    )
    
    print(f"Done, results saved in {str(folder)}")
    
    
if __name__ == "__main__":
    main()