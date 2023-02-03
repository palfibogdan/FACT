# Reproducing "[Online certification of preference-based fairness for personalized recommender systems](https://arxiv.org/pdf/2104.14527.pdf)"
--- 
## Installation
--- 
It is suggested to use a virtual environment to run this project,
such as [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html). The following commands work with 
(Mini)Conda.

Assuming you are in the root directory of the git project, to install the MLRC_OCEF environment run:
```
conda env create -f env.yml
```

(Optional) Install development dependencies, such as Jupyter Notebook/Lab:
```
conda env update -f env.dev.yml
```
---
## Running the results notebook
---
Follow these steps to run a Jupyter notebook presenting the results
reproduced from the original paper. These command work on Linux; 
follow a similar process on Windows or macOS.

1. Unzip the pretrained models for experiment 5.1 "Sources of envy from model misspecification" in the root directory of the git project.
   ```
   unzip models_and_hyperparams.zip
   ```
   Make sure to place the contents of the archive - `movielens-1m` and 
   `lastfm` - at the root of the git project.
2. Activate the `MLRC_OCEF` environment (e.g. with Conda):
   ```
   conda activate MLRC_OCEF
   ```
3. Start a notebook from inside the `MLRC_OCEF` Python package, and 
   evaluate all the cells:
   ```
   cd MLRC_OCEF
   jupyter lab
   ```
---
## Running the project
---
The main entry point for our reproduction study is `MLRC_OCEF/main.py`.
To see the different options accepted by the program, assuming you 
are at the root of the git project, run:
```
python MLRC_OCEF/main.py -h
```
Our reproduction study is configurable, and the above command will 
present all the various running options.

### Bandit experiment
To reproduce the synthetic bandit experiment from the original 
paper, run:
```
python MLRC_OCEF/main.py --experiment bandit-synthetic
```
If the project's root contains the folder `results_ocef`, precomputed 
cost and duration files will be loaded from there, and will produce 
a graph in `results_ocef/ocef.png`. To run the bandit experiment from 
scratch, rename or delete the `results_ocef` folder and run the above 
command.

### Envy from model misspecification
To fully reproduce the experiments concerning envy from model misspecification, make sure there are no folders `movielens-1m` and 
`lastfm` at the root of the git project. Then, run:
```
python MLRC_OCEF/main.py --experiment envy-misspecification
```

To reproduce all the additional plots in `results_envy`, run the `MLRC_OCEF/envy_experiments.py` script; new plots will be saved to `results_envy`. Then, you can use the configurations described in that script from inside `MLRC_OCEF/results_notebook.ipynb` to visualize the plots interactively. 

Alternatively, you can copy any of those configurations in `MLRC_OCEF/results_notebook.ipynb` and evaluate the notebook to produce the corresponding plot.

---
## References
---
To refer to the original paper, please cite:
```
@article{online_cert,
  author    = {Virginie Do and
	       Sam Corbett{-}Davies and
	       Jamal Atif and
	       Nicolas Usunier},
  title     = {Online certification of preference-based fairness for personalized
	       recommender systems},
  journal   = {CoRR},
  volume    = {abs/2104.14527},
  year      = {2021},
  url       = {https://arxiv.org/abs/2104.14527},
  eprinttype = {arXiv},
  eprint    = {2104.14527},
  timestamp = {Tue, 04 May 2021 15:12:43 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2104-14527.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```