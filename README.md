## dMaSIF - Fast end-to-end learning on protein surfaces
![Method overview](overview.PNG)

## Abstract

Proteins’ biological functions are defined by the geometric
and chemical structure of their 3D molecular surfaces.
Recent works have shown that geometric deep learning can
be used on mesh-based representations of proteins to identify
potential functional sites, such as binding targets for
potential drugs. Unfortunately though, the use of meshes as
the underlying representation for protein structure has multiple
drawbacks including the need to pre-compute the input
features and mesh connectivities. This becomes a bottleneck
for many important tasks in protein science.

In this paper, we present a new framework for deep
learning on protein structures that addresses these limitations.
Among the key advantages of our method are the computation
and sampling of the molecular surface on-the-fly
from the underlying atomic point cloud and a novel efficient
geometric convolutional layer. As a result, we are able to
process large collections of proteins in an end-to-end fashion,
taking as the sole input the raw 3D coordinates and
chemical types of their atoms, eliminating the need for any
hand-crafted pre-computed features.

To showcase the performance of our approach, we test it
on two tasks in the field of protein structural bioinformatics:
the identification of interaction sites and the prediction
of protein-protein interactions. On both tasks, we achieve
state-of-the-art performance with much faster run times and
fewer parameters than previous models. These results will
considerably ease the deployment of deep learning methods
in protein science and open the door for end-to-end differentiable
approaches in protein modeling tasks such as function
prediction and design.

## Hardware requirements

Models have been trained on either a single NVIDIA RTX 2080 Ti or a single Tesla V100 GPU. Time and memory benchmarks were performed on a single Tesla V100.

## Software prerequisites 

Scripts have been tested using the following two sets of core dependencies:

| Dependency | First Option  | Second Option | Updated Version |
| ------------- | ------------- | ------------- | ------------- |
| GCC | 7.5.0 | 8.4.0 | 9.2.0 |
| CMAKE | 3.10.2 | 3.16.5 | 3.22.2 |
| CUDA | 10.0.130 | 10.2.89  | 11.7 |
| cuDNN | 7.6.4.38  | 7.6.5.32  | 7.6.x |
| Python | 3.6.9  | 3.7.7  | 3.8.16 |
| PyTorch | 1.4.0  | 1.6.0  | 1.13.1 |
| PyKeops | 1.4  | 1.4.1  | 2.1.1 |
| PyTorch Geometric | 1.5.0  | 1.6.1  | 2.2.0 |


## Code overview


Usage:
- In order to **train models**, run `main_training.py` with the appropriate flags. 
Available flags and their descriptions can be found in `Arguments.py`.

- The command line options needed to reproduce the **benchmarks** can be found in `benchmark_scripts/`.

- To make **inference** on the testing set using pretrained models, use `main_inference.py` with the flags that were used for training the models. 
Note that the `--experiment_name flag` should be modified to specify the training epoch to use.

Implementation:
- Our **surface generation** algorithm, **curvature** estimation method and **quasi-geodesic convolutions** are implemented in `geometry_processing.py`.

- The **definition of the neural network** along with surface and input features can be found in `model.py`. The convolutional layers are implemented in `benchmark_models.py`.

- The scripts used to **generate the figures** of the paper can be found in `data_analysis/`.


## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>.

## Reference

Sverrisson, F., Feydy, J., Correia, B. E., & Bronstein, M. M. (2020). Fast end-to-end learning on protein surfaces. [bioRxiv](https://www.biorxiv.org/content/10.1101/2020.12.28.424589v1).