# Location-Aware Generative Adversarial Networks (LAGAN) for Physics Synthesis
This repository contains all the code used in L. de Oliveira ([@lukedeo](https://github.com/lukedeo)), M. Paganini ([@mickypaganini](https://github.com/mickypaganini)), B. Nachman ([@bnachman](https://github.com/bnachman)), _Learning Particle Physics by Example: Location-Aware Generative Adversarial Networks for Physics Synthesis_ [[`arXiv:1701.05927`](https://arxiv.org/abs/1701.05927)]

## Citations
You are more than welcome to use the open data and open-source software provided here for any of your projects, but we kindly ask you that you please cite them using the DOIs provided below:

| Asset  | Location |
| ------------- | ------------- |
| Source Code (this repository) | [![DOI](https://zenodo.org/badge/74294060.svg)](https://zenodo.org/badge/latestdoi/74294060) |
| Data (Pythia images) | [![DOI](https://zenodo.org/badge/DOI/10.17632/4r4v785rgx.1.svg)](https://doi.org/10.17632/4r4v785rgx.1) |
| Model Weights  | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.400706.svg)](https://doi.org/10.5281/zenodo.400706) |
| Docker image (generation) | [![](https://images.microbadger.com/badges/version/lukedeo/ji.svg)](https://hub.docker.com/r/lukedeo/ji/) |

If you're using ideas or methods discussed in the paper, with or without using the software, please cite:
```
@article{lagan,
      author         = "de Oliveira, Luke and Paganini, Michela and Nachman, Benjamin",
      title          = "{Learning Particle Physics by Example: Location-Aware
                        Generative Adversarial Networks for Physics Synthesis}",
      year           = "2017",
      eprint         = "1701.05927",
      archivePrefix  = "arXiv",
      primaryClass   = "stat.ML",
      SLACcitation   = "%%CITATION = ARXIV:1701.05927;%%"
}
```

## Getting Started

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
