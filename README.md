[![License](https://img.shields.io/badge/License-Apache-red)](https://www.nature.com/articles/s41467-021-25756-4)
# plmDCA
pseudo-likelihood Maximization Direct Coupling Analysis (plmDCA) 2.0

## Overview
This package is the GPU-accelerated version of the original code that can be found at [PlmDCA.jl](https://github.com/pagnani/PlmDCA.jl.git). However, this implementation focuses more on the generation rather than contact prediction. It also aims at providing a user-friendly command line interface for training and sampling from a pseudo-likelihood DCA model.

## Installation
During the installation, `plmDCA` will also install `adabmDCA` and all its dependencies.

To install, clone this repository:
```bash
git clone https://github.com/spqb/arDCA.git
cd arDCA
python -m pip install .
```
## Using the package
We provide a [Colab notebook](https://colab.research.google.com/drive/1z0z0-CT6iW6g2lZEYfnEVsyVfcexfrHX?usp=sharing) where it is shown hot to train and sample a `plmDCA` model.

Alternatively, one can install the package locally and run from the command line one of the two implemented routines:

### Train plmDCA from the command line
Once installed, you can launch the package routing by using the command `plmDCA`. All the training options can be listed via
```bash
plmDCA train -h
```
To launch a training with default arguments, use
```bash
plmDCA train -d <path_data> -o <output_folder> -l <label>
```
where `path_data` is the path to the input multi-sequence alignment in [fasta](https://en.wikipedia.org/wiki/FASTA_format) format and `label` is an identifier for the output files. The parameters of the trained model are saved in the file `output_folder/<label>_params.pth`, and can be easily loaded afterwrds using the Pytorch methods.

By default, the program assumes that the input data are protein sequences. If you want to use RNA sequences, you should use the argument `--alphabet rna`.

> [!WARNING]
> Depending on the dataset, the default regularization parameters `reg_h` and `reg_J` may not work properly. If the training does not converge or the model's generation capabilities are poor, you may want to increase these values.

> [!NOTE]
> By default, the training routine returns the non-reciprocal (i.e. asymmetric) version of the model's couplings. To obtain the reciprocal (symmetric) version, add the flag `--reciprocal` to the script's arguments.

### Sample plmDCA from the command line
To generate new sequences using the command line, the minimal input command is
```bash
plmDCA sample -p <path_params> -d <path_data> -o <output_folder> -l <label> --ngen <num_sequences>
```
where `num_sequences` is the number of sequences to be generated. The output sequences will be saved in fasta format at `output_folder/<label>_samples.fasta`. Also, a `csv` log file containing the one and two-sites Pearson correlation coefficients between the generated samples and the data as a function of the number of sweeps is returned. 

> [!NOTE]
> By default, the script will take the first sequence of the input MSA as the wild-type to start the sampling from. To use a random intialization, add the flag `--init_random`.

## License
This package is open-sourced under the Apache License 2.0.

## Citation
If you use this package in your research, please cite
> Trinquier, J., Uguzzoni, G., Pagnani, A. et al. Efficient generative modeling of protein sequences using simple autoregressive models. Nat Commun 12, 5800 (2021).

