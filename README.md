* Notes about the paper, details about implementation, etc are 
  [here](notes/paper_notes.md).

# Environment installation
--- 
It is suggested to use a virtual environment to run this project,
such as [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or the faster implementation with almost the 
same interface [Micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html). The following commands work with 
(Mini)Conda; don't use the `env` command with (Micro)Mamba.

Install FACT environment:
```
conda env create -f env.yml
```

(Optional) Install development dependencies:
```
conda env update -f env.dev.yml
```