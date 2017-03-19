# Location-Aware Generative Adversarial Networks (LAGAN) for Physics Synthesis

| Asset  | Location |
| ------------- | ------------- |
| Data (Pythia images) | [![DOI](https://zenodo.org/badge/DOI/10.17632/4r4v785rgx.1.svg)](https://doi.org/10.17632/4r4v785rgx.1) |
| Model Weights  | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.400706.svg)](https://doi.org/10.5281/zenodo.400706) |
| Docker image (generation) | [![](https://images.microbadger.com/badges/version/lukedeo/ji.svg)](https://hub.docker.com/r/lukedeo/ji/) |

This repository contains all the code used in L. de Oliveira ([@lukedeo](https://github.com/lukedeo)), M. Paganini ([@mickypaganini](https://github.com/mickypaganini)), B. Nachman ([@bnachman](https://github.com/bnachman)), _Learning Particle Physics by Example: Location-Aware Generative Adversarial Networks for Physics Synthesis_ [[`arXiv:1701.05927`](https://arxiv.org/abs/1701.05927)]

To clone everything necessary, you'll need to run `git clone --recursive https://github.com/lukedeo/adversarial-jets` to fetch all the submodules (you can add a `-j6` or some other number to launch concurrent clones).

This repository is structured as such:
* [generation](#generation) 
* [models](#models)
* [analysis](#analysis)

### Generation
[TODO: allow for preprocessing in Docker]
This folder links to the submodule used for generating Pythia images. 

### Models
This folder contains the Keras models used for training the LAGAN seen in the paper. By running `python train.py -h` from this folder, you should see all available options for running the training, as well as how to find / download the data required.

### Analysis
[TODO: update filenames, download links, etc.]
This folder contains a jupyter nootbook that will guide you through the production of the plots that appear in the paper. You will be able to reproduce them and modify them as you wish using our trained models and open datasets, or reuse the plotting functions to visualize the performance of your own LAGAN.

Simply run: ``jupyter notebook plots.ipynb``
