# Rate-Splitting for Intelligent Reflecting Surface-Aided Multiuser VR Streaming

This is the PyTorch implementation for the deep deterministic policy gradient with imitation learning (Deep-GRAIL) algorithm, with a specific application to solve the sum-rate maximization problem in rate-splitting multiuser systems. The code includes the implmentations of the learning algorithm, deep neural network models, system model for the rate-spliting multiuser systems, as well as a conventional optimization algorithm. If our codes and data are helpful to your research, please kindly cite [the paper]([https://ieeexplore.ieee.org/document/10032264]). The application in the multiuser VR streaming systems will be updated upon final paper acceptance. Please check this page for updates.

### Prerequisites

The following libraries are required for this code base. We recommend to use the same versions as listed.
* Python v3.7.9
* PyTorch v1.9.0
* Numpy v1.19.1
* Matlab Engine for Python (R2020a) (Only required for generating the demonstration replay. [How to install Matlab Engine for Python](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html))

### Usage
1. Generate the demonstration replay by running:
```
python demonstration_gen.py
```
Running the script will generate the demonstration replay and store it as NumPy array file (.npy) with names `replay_*.npy`. 

You can skip this step by using the pre-generated demonstration replay (together with the experience replay), which can be downloaded [here](https://drive.google.com/file/d/1PCTX1Li6Gow6G3ij0vK2dCuPVHBV2FzZ/view?usp=share_link) (File size ~1.5 GB).

2. Start the training algorithm by running:
```
python main.py
```

### Structures
* `main.py`: Main training loop and the implementation of Markov decision process (MDP).
* `DeepGRAIL.py`: The learning algorithm.
* `networks.py`: The implementation of the deep neural networks.
* `utils.py`: The implementation of the replay, including adding, loading, saving, and sampling of transition tuples.
* `demonstration_gen.py`: Method for generating the demonstration replay. The MDP implemented here needs to be the same as `main.py`.
* `opt_algo.m`: Matlab script for the conventional optimization algorithm (e.g., alternating optimization (AO) algorithm.

### Bibtex

```
@ARTICLE{rui2023jsac,
  author={Huang, Rui and Wong, Vincent W.S. and Schober, Robert},
  journal={IEEE Journal on Selected Areas in Communications}, 
  title={Rate-Splitting for Intelligent Reflecting Surface-Aided Multiuser VR Streaming}, 
  year={2023},
  volume={41},
  number={5},
  pages={1516-1535},
  doi={10.1109/JSAC.2023.3240710}}
```
