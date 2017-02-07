# Location-Aware Generative Adversarial Networks (LAGAN) for Physics Synthesis

| Asset  | Location |
| ------------- | ------------- |
| Data (Pythia images) | [![DOI](https://zenodo.org/badge/DOI/10.17632/4r4v785rgx.1.svg)](https://doi.org/10.17632/4r4v785rgx.1) |
| Model Weights  |  |
| Docker image (generation) | [![](https://images.microbadger.com/badges/version/lukedeo/ji.svg)](https://hub.docker.com/r/lukedeo/ji/) |
| Docker image (training) |  |

This repository contains all the code used in L. de Oliveira ([@lukedeo](https://github.com/lukedeo)), M. Paganini ([@mickypaganini](https://github.com/mickypaganini)), B. Nachman ([@bnachman](https://github.com/bnachman)), _Learning Particle Physics by Example: Location-Aware Generative Adversarial Networks for Physics Synthesis_ [[arXiv:1701.05927](https://arxiv.org/abs/1701.05927)]

This repository is structured as such:
* [generation](#generation) 
* [models](#models)
* [analysis](#analysis)

### Generation
This folder links to the submodule used for generating Pythia images.

### Models
This folder contains the Keras models used for training various versions of the LAGAN. 

### Analysis
This folder contains a jupyter nootbook that will guide you through the production of the plots that appear in the paper. You will be able to reproduce them and modify them as you wish using our trained models and open datasets, or reuse the plotting functions to visualize the performance of your own LAGAN.

Simply run: ``jupyter notebook plots.ipynb``
