# Deep-GRAIL Algorithm: Deep Deterministic Policy Gradient with Imitation Learning

<!-- ABOUT THE PROJECT -->
## About The Project

This is the code base for the Deep-GRAIL algorithm, with a specific application to solve the sum-rate maximization problem in rate-splitting multiuser systems. The code includes the implmentations of the learning algorithm, deep neural network models, system model for the rate-spliting multiuser systems, as well as a conventional optimization algorithm. If you use our code or data please cite [the paper](https://arxiv.org/abs/2210.12191).

### Prerequisites

The following libraries are required for this code base. We recommend to use the same versions as listed.
* Python v3.7.9
* PyTorch v1.9.0
* Numpy v1.19.1
* Matlab Engine for Python (R2020a) (Only required for generating the demonstration replay, see [how to install Matlab Engine](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html))

<!-- USAGE EXAMPLES -->

### Usage
1. Generate the demonstration replay by running:
```
python demonstration_gen.py
```
Running the script will generate the demonstration replay and store it as NumPy array file (.npy) with names `replay_*.npy`. You may skip this process by using our pre-generated demonstration replay, which can be downloaded here.

2. Start the training algorithm by running:
```
python main.py
``` 

### Bibtex

```
@article{rui2022rate,
    title={Rate-Splitting for Intelligent Reflecting Surface-Aided Multiuser {VR} Streaming},
   	author={Huang, Rui and  W.S. Wong, Vincent and Schober, Robert},
    year={2022},
    month = {Oct.},
    journal={arXiv preprint arXiv:2210.12191},
}
```
