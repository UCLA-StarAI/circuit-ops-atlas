# circuit-ops-atlas

Code of the NeurIPS 2021 paper "A Compositional Atlas of Tractable Circuit Operations for Probabilistic Inference"

## Run the experiments

1. Generate the PCs using Strudel:

```
sh strudel.sh
```

All generated PCs are stored in `pcs/`.

2. Follow the notebook `experiments.ipynb` to reproduce experiment results in the paper.

## Citation

To cite this paper, please use

```
@inproceedings{VergariNeurIPS21,
    title   = {A Compositional Atlas of Tractable Circuit Operations for Probabilistic Inference},
    author = {Vergari, Antonio and Choi, YooJung and Liu, Anji and Teso, Stefano and Van den Broeck, Guy},
    booktitle = {35th Conference on Neural Information Processing Systems (NeurIPS 2021)},
    month   = {dec},
    year    = {2021}
}
```
